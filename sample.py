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
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner
from dataset import imagenet
from i2sb import ckpt_util


from dataset import load_data
from evaluations.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim_measure
from torchmetrics.aggregation import RunningMean


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

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
    else:
        clean_img, y = out
        mask = None
        corrupt_img = corrupt_method(clean_img.to(opt.device))
        x1 = corrupt_img.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return corrupt_img, x1, mask, cond, y

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, opt.ckpt_path)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # Custom dataloader
    train_dataset, val_dataset = load_data(
        data_dir=opt.dataset_dir,
        dataset=opt.dataset,
        batch_size=opt.batch_size,
        image_size=opt.data_image_size,
        num_channels=opt.data_image_channels,
        num_workers=1,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)
    runner.net.eval()

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # Compute FID, MSE, MAE, SSIM
    # ---------------------------------------
    print("Computing FID of model and L1 score")
    # sample 50,000 images for computeing fid against train set
    print("\nGenerating fid samples...", flush=True)

    num_batches = 50000//len(train_dataset)
    for _ in range(num_batches):
        for i, (x1_batch, x0_batch, _) in enumerate(train_dataset):
            x1_batch = preprocess(x1_batch)
            x0_batch = preprocess(x0_batch)
            mask = None
            y = None
            cond = x1_batch if opt.cond_x1 else None
            x1 = x0_batch

            xs, pred_x0s = runner.ddpm_sampling(
                opt, x1_batch, mask=mask, cond=x0_batch, clip_denoise=True, verbose=opt.global_rank==0
            )
            print(xs.shape)
            xs = xs[:, 0, ...]
            xs = (xs+1.)/2.
            xs = xs.cpu().detach()
            xs = xs.permute(0,2,3,1).numpy()
            for j in range(len(xs)):
                plt.imsave('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/fid_samples/image_{}_{}_{}.png'.format(i*opt.batch_size+j), xs[j,...]) # When generating multichannel data
                # plt.imsave('fid/{}/image_{}_{}_{}.JPEG'.format("fid_samples", i*batch_size+j), samples[j,:,:,0], cmap='gray') # When generating single channel data
    dist.barrier()
    print("\nfinished generating fid samples.", flush=True)

    print("\ncomputing fid...")
    dir2ref = "/home/sszabados/datasets/lysto64_random_crop_pix2pix/B/val/"
    dir2gen = "/home/sszabados/checkpoints/i2sb/lysto64_random_crop/fid_samples/"
    fid_value = 0
    try:
        fid_value = calculate_fid_given_paths(
            paths = [dir2ref, dir2gen],
            batch_size = 128,
            device = "cuda:1",
            img_size = 256,
            dims = 2048,
            num_workers = 1,
            eqv = 'D4' 
        )
    except ValueError:
        fid_value = np.inf
    # Incrementally save fids after each epoch
    os.makedirs('/u6/sszabado/checkpoints/i2sb/lysto64_random_crop_ddbm/metrics/', exist_ok=True)
    np.save('/u6/sszabado/checkpoints/i2sb/lysto64_random_crop_ddbm/metrics/fid.npy', np.array(fid_value))
    print(f"FID: {fid_value}")

    print("Computing MSE, MSA, SSIM, loss...")

    print(len(val_dataset))

    ref_samples = []
    samples = []

    for i, (x1_batch, x0_batch, _) in enumerate(val_dataset):
        x1_batch = preprocess(x1_batch)
        x0_batch = preprocess(x0_batch)
        mask = None
        y = None
        cond = x1_batch if opt.cond_x1 else None
        x1 = x0_batch
        
        xs, pred_x0s = runner.ddpm_sampling(
                opt, x1_batch, mask=mask, cond=x0_batch, clip_denoise=True, verbose=opt.global_rank==0
            )
        xs = xs[:, 0, ...]
        samples.append(runner.all_cat_cpu(opt, log, xs))
        ref_samples.append(runner.all_cat_cpu(opt, log, x0_batch))
        if i*opt.batch_size > 1500:
            break

    dist.barrier()
    samples = torch.cat(samples, dim=0)
    ref_samples = torch.cat(ref_samples, dim=0)

    mae_T = RunningMean()
    mse_T = RunningMean()
    ssim_T = ssim_measure(data_range=(-1,1))
    # ssim_score = ssim_T(preds=samples, target=ref_samples).item()
    samples_len = len(samples)
    num_batches = samples_len//100
    for i in range(0, num_batches):
        s = i*100
        e = min(s+100, samples_len)
        mse_T.update(torch.mean((samples[s:e]-ref_samples[s:e])**2, dim=(1,2,3)))
        mae_T.update(torch.mean(torch.abs(samples[s:e]-ref_samples[s:e]), dim=(1,2,3)))
        ssim_T.update(samples[s:e], ref_samples[s:e])
    mse = mse_T.compute().item()
    mae = mae_T.compute().item()
    ssim_score = ssim_T.compute().item()
    dist.barrier()
    
    os.makedirs('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/', exist_ok=True)
    np.save('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/mae.npy', np.array(mae))
    os.makedirs('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/', exist_ok=True)
    np.save('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/ssim.npy', np.array(ssim_score))

    print(f"MAE: {mae}, SSIM: {ssim_score}")

    del runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image_size",     type=int,  default=256)
    parser.add_argument("--data_image_size", type=int,  default=256)
    parser.add_argument("--data_image_channels", type=int, default=3)
    parser.add_argument("--dataset_dir",    type=Path, default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--dataset",        type=str,   default="")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--batch_size",     type=int,  default=32)
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("--ot-ode",         action="store_true",             help="use OT-ODE model")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--log_dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")

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
