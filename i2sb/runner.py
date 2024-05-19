# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim_measure
from torchmetrics.aggregation import RunningMean

from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision
import torchvision.utils as tu


import distributed_util as dist_util

from . import util
from .network import Image256Net
from .diffusion import Diffusion

from ipdb import set_trace as debug

from evaluations.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt


def preprocess(x):
    """
    Preprocessing function taken from train_util.py
    """    
    if x.shape[1] == 3:
        x =  2.*x - 1.
    return x

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        self.resume_it = 0

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")
            # Set resume step
            self.resume_it = int(os.path.splitext(os.path.basename(opt.load))[0])
        
        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

        # Evaluation metrics 
        self.MAE = []
        self.MSE = []
        self.fids = []
        self.SSIM = []

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()

        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)

        it = 0
        if opt.ckpt is not None and it < self.resume_it:
            it = self.resume_it + 1
        for _, (x0_batch, x1_batch, _) in enumerate(train_dataset):
            optimizer.zero_grad()

            x0_batch = preprocess(x0_batch)
            x1_batch = preprocess(x1_batch)

            # Iterate over microbatches
            for i in range(n_inner_loop):
                # ===== sample boundary pair =====
                # x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)
                x0 = x0_batch[(i*opt.microbatch):(i*opt.microbatch+opt.microbatch)].to(opt.device)
                x1 = x1_batch[(i*opt.microbatch):(i*opt.microbatch+opt.microbatch)].to(opt.device)
                y = None
                mask = None
                cond = x1 if opt.cond_x1 else None

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                loss = F.mse_loss(pred, label)
                loss.backward()

            # Update loss after micobatches have completed
            optimizer.step()
            ema.update()
            it += 1

            del x0_batch
            del x1_batch
            torch.cuda.empty_cache()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / f"{it}.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it == 500 or (it > 0 and it % 5000 == 0): # 0, 0.5k, 3k, 6k 9k
                net.eval()
                self.evaluation(opt, it, train_dataset, val_dataset)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, train_dataset, val_dataset):

        log = self.log
        log.info(f"========== Evaluating model : iter={it} ==========")
        torch.cuda.empty_cache()

        # Generate training sample
        # ---------------------------------------
        num_samples = 5

        test_batch, test_cond, _ = next(iter(val_dataset))
        batch_size = len(test_batch)

        if num_samples > batch_size:
            num_samples = batch_size

        test_batch = preprocess(test_batch[0:num_samples])
        test_cond = preprocess(test_cond[0:num_samples])

        img_clean = test_batch
        img_corrupt = test_cond
        mask = None
        y = None
        cond = img_clean if opt.cond_x1 else None
        x1 = img_corrupt

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, mask=mask, cond=cond, clip_denoise=True, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")

        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        # y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        # assert y.shape == (batch,)

        img_recon = xs[:, 0, ...]
        pred_x0s = pred_x0s[:, 0,...]

        mae_T = RunningMean().to(opt.device)
        mse_T = RunningMean().to(opt.device)
        ssim_T = ssim_measure(data_range=(-1,1)).to(opt.device)
        mse_T.update(torch.mean((img_recon.to(opt.device)-img_clean.to(opt.device))**2, dim=(1,2,3)))
        mae_T.update(torch.mean(torch.abs(img_recon.to(opt.device)-img_clean.to(opt.device)), dim=(1,2,3)))
        ssim_T.update(img_recon.to(opt.device), img_clean.to(opt.device))
        mse = mse_T.compute().item()
        mae = mae_T.compute().item()
        ssim_score = ssim_T.compute().item()
        print(f"MSE: {mse}, MAE: {mae}, SSIM: {ssim_score}")

        log.info(f"Generated recon trajectories: size={xs.shape}")
        log.info("Logging images ...")

        grid_img = torchvision.utils.make_grid(img_clean, nrow=num_samples, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, 'tmp_imgs/clean_{}_{}_{}.png'.format(it, mae, ssim_score))
        grid_img = torchvision.utils.make_grid(img_corrupt, nrow=num_samples, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, 'tmp_imgs/currupt_{}_{}_{}.png'.format(it, mae, ssim_score))
        grid_img = torchvision.utils.make_grid(img_recon, nrow=num_samples, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, 'tmp_imgs/img_recon_{}_{}_{}.png'.format(it, mae, ssim_score))
        grid_img = torchvision.utils.make_grid(pred_x0s, nrow=num_samples, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid_img, 'tmp_imgs/pred_x0s_{}_{}_{}.png'.format(it, mae, ssim_score))
   
        del mae_T
        del mse_T
        del ssim_T

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
                xs, pred_x0s = self.ddpm_sampling(
                    opt, x1_batch, mask=mask, cond=x0_batch, clip_denoise=True, verbose=opt.global_rank==0
                )
                xs = xs[:, 0, ...]
                xs = (xs+1.)/2.
                xs = xs.cpu().detach()
                xs = xs.permute(0,2,3,1).numpy()
                for j in range(len(xs)):
                    plt.imsave('/u6/sszabado/checkpoints/i2sb/lysto64_random_crop_ddbm/fid_samples/image_{}_{}_{}.png'.format(i*opt.batch_size+j), xs[j]) # When generating multichannel data
                    # plt.imsave('fid/{}/image_{}_{}_{}.JPEG'.format("fid_samples", i*batch_size+j), samples[j,:,:,0], cmap='gray') # When generating single channel data
        print("\nfinished generating fid samples.", flush=True)

        print("\ncomputing fid...")
        dir2ref = "/share/yaoliang/datasets/lysto64_random_crop_pix2pix/B/val"
        dir2gen = "/u6/sszabado/checkpoints/i2sb/lysto64_random_crop_ddbm/fid_samples/"
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
        self.fids.append([it,fid_value])
        # Incrementally save fids after each epoch
        os.makedirs('/u6/sszabado/checkpoints/i2sb/lysto64_random_crop_ddbm/metrics/', exist_ok=True)
        np.save('/u6/sszabado/checkpoints/i2sb/lysto64_random_crop_ddbm/metrics/fid.npy', np.array(self.fids))
        print(f"FID: {fid_value}")

        print("Computing MSE, MSA, SSIM, loss...")

        print(len(val_dataset))

        ref_samples = []
        samples = []
        for i, (x1_batch, x0_batch, _) in enumerate(val_dataset):
            x1_batch = preprocess(x1_batch)
            x0_batch = preprocess(x0_batch)
            xs, pred_x0s = self.ddpm_sampling(
                    opt, x1_batch, mask=mask, cond=x0_batch, clip_denoise=True, verbose=opt.global_rank==0
                )
            xs = xs[:, 0, ...]
            samples.append(all_cat_cpu(opt, log, xs))
            ref_samples.append(all_cat_cpu(opt, log, x0_batch))
            if i*opt.batch_size > 1500:
                break
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
        
        self.MSE.append([it, mse])
        self.MAE.append([it, mae])
        self.SSIM.append([it, ssim_score])
        os.makedirs('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/', exist_ok=True)
        np.save('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/mse.npy', np.array(self.MSE))
        os.makedirs('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/', exist_ok=True)
        np.save('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/mae.npy', np.array(self.MAE))
        os.makedirs('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/', exist_ok=True)
        np.save('/home/sszabados/checkpoints/i2sb/lysto64_random_crop/metrics/ssim.npy', np.array(self.SSIM))

        print(f"MSE: {mse}, MAE: {mae}, SSIM: {ssim_score}")


        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
