# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner

from dataset import load_data
from evaluations.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim_measure
from torchmetrics.aggregation import RunningMean
import itertools


RESULT_DIR = Path("results")

def preprocess(x):
    """
    Preprocessing function taken from train_util.py
    """    
    if x.shape[1] == 3:
        x =  2.*x - 1.
    return x

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # Custom dataloader
    train_dataset, val_dataset = load_data(
        data_dir=opt.dataset_dir,
        dataset=opt.dataset,
        batch_size=opt.batch_size,
        image_size=opt.data_image_size,
        num_channels=opt.data_image_channels,
        num_workers=1,
    )

    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training
    # build runner
    if opt.ckpt is not None:
        ckpt_file = opt.ckpt_path / opt.ckpt 
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    runner = Runner(opt, log)
    runner.net.eval()

    # Compute FID, MSE, MAE, SSIM
    # ---------------------------------------
    print("\nGenerating fid samples...", flush=True)

    num_batches = 1
    if opt.num_samples > len(train_dataset):
        num_batches = opt.num_samples//len(train_dataset)

    if opt.partition is not None:
        for i, (x0_batch, x1_batch, _) in enumerate(itertools.islice(train_dataset, opt.partition*opt.num_samples, None)):
            x1_batch = preprocess(x1_batch)
            x0_batch = preprocess(x0_batch)
            mask = None

            img_clean = x0_batch
            img_corrupt = x1_batch
            mask = None
            y = None
            cond = img_clean if opt.cond_x1 else None
            x1 = img_corrupt

            xs, pred_x0s = runner.ddpm_sampling(
                opt, x1, mask=mask, cond=cond, clip_denoise=True, verbose=opt.global_rank==0
            )

            xs = xs[:, 0, ...]
            xs = (xs+1.)/2.
            xs = xs.clamp(0,1)
            xs = xs.cpu().detach()
            xs = xs.permute(0,2,3,1).numpy()
            for j in range(len(xs)):
                plt.imsave(f'/ssd005/projects/watml/szabados/checkpoints/I2SB/lysto64_random_crop/fid_samples/image_{opt.partition}_{i*opt.batch_size+j}.png', xs[j]) # When generating multichannel data
            print("Generated: "+str(i*opt.batch_size)+" many samples.")
            if i*opt.batch_size > opt.num_samples:
                print("Finished generating samples.")
                break
        dist.barrier()
    else:
        for k in range(num_batches):
            for i, (x0_batch, x1_batch, _) in enumerate(train_dataset):
                x1_batch = preprocess(x1_batch)
                x0_batch = preprocess(x0_batch)
                mask = None

                img_clean = x0_batch
                img_corrupt = x1_batch
                mask = None
                y = None
                cond = img_clean if opt.cond_x1 else None
                x1 = img_corrupt

                xs, pred_x0s = runner.ddpm_sampling(
                    opt, x1, mask=mask, cond=cond, clip_denoise=True, verbose=opt.global_rank==0
                )
                
                xs = xs[:, 0, ...]
                xs = (xs+1.)/2.
                xs = xs.clamp(0,1)
                xs = xs.cpu().detach()
                xs = xs.permute(0,2,3,1).numpy()
                for j in range(len(xs)):
                    plt.imsave(os.path.join(opt.save_dir,f'/image_{opt.partition}_{i*opt.batch_size+j}.png'), xs[j]) # When generating multichannel data
        dist.barrier()
    print("\nfinished generating fid samples.", flush=True)

    del runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")
    parser.add_argument("--corrupt",        type=str,   default=None,        help="restoration task")
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--beta-max",       type=float, default=0.3,         help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.99)
    
    # data
    parser.add_argument("--image_size",     type=int,  default=256)
    parser.add_argument("--data_image_size", type=int,  default=256)
    parser.add_argument("--data_image_channels", type=int, default=3)
    parser.add_argument("--dataset_dir",    type=Path, default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--dataset",        type=str,   default="")
    parser.add_argument("--partition",      type=int,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--interval",       type=int,  default=1000,        help="number of interval")
    parser.add_argument("--num_samples",    type=int,  default=1000)
    parser.add_argument("--batch_size",     type=int,  default=32)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="resumed checkpoint name")
    parser.add_argument("--ckpt_path",      type=Path, default=None)
    parser.add_argument("--ot-ode",         action="store_true",            help="use OT-ODE model")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--log_dir",        type=Path,  default=".log",     help="path to log std outputs and writer data")
    parser.add_argument("--cond-x1",        action="store_true",            help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",   action="store_true",            help="add noise to conditional network")
    parser.add_argument("--save_dir",       type=str, default="")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    # download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
