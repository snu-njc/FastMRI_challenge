import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

import fastmri as fastmri
import config_file
import losses
import sampling
import sde_lib
import controllable_generation
from models import ddpm, ncsnv2, ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector)

def train_epoch(args, epoch, train_step_fn, data_loader, state):
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        
        b, c, h, w, two = kspace.shape
        assert two == 2
        kspace = kspace.view(b*c, h, w, 2).permute(0, 3, 1, 2)
        kspace = kspace[:, :, :, w//2 - 192 : w//2 + 192]
        kspace = kspace.cuda(non_blocking=True)
        
        loss = train_step_fn(state, kspace)
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, score_model, pc_inpainter, data_loader):
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start_iter = start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            
            b, c, h, w, two = kspace.shape
            assert two == 2
            kspace = kspace.view(b*c, h, w, 2).permute(0, 3, 1, 2)
            kspace = kspace[:, :, :, w//2 - 192 : w//2 + 192]
            mask = mask.view(1, 1, 1, -1)
            mask = mask[:, :, :, w//2 - 192 : w//2 + 192]
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            output = pc_inpainter(score_model, kspace, mask)
            output = output.permute(0, 2, 3, 1).view(b, c, h, 384, 2)
            result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(output)), dim=1)
            height = result.shape[-2]
            width = result.shape[-1]
            output = result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]

            print(
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
            if iter >= 10:
                break

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    config = config_file.get_config()
    
    print("defining models")
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    print("defining sde")
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
    
    print("defining step functions")
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)
    
    print("defining pc inpainter")
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    snr = 0.16
    n_steps = 1
    probability_flow = False
    pc_inpainter = controllable_generation.get_pc_inpainter(sde,
                                                            predictor, corrector,
                                                            snr=snr,
                                                            n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous,
                                                            denoise=True)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, train_step_fn, train_loader, state)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, score_model, pc_inpainter, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)


        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

#         save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
