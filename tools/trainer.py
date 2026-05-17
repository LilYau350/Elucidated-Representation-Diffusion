import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tools import dist_util, logger
import os
import torch.distributed as dist
from tools.align_utils import initialize_encoders, get_feature
from collections import OrderedDict
from contextlib import nullcontext


def ema(source, target, decay):
    with torch.no_grad():
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay + source_dict[key].data * (1 - decay))


def sample_from_latent(latent, latent_scale=1.):
    mean, std = torch.chunk(latent, 2, dim=1)
    latent_samples = mean + std * torch.randn_like(mean)
    latent_samples = latent_samples * latent_scale 
    return latent_samples 


class Trainer:
    def __init__(self, args, device, model, ema_model, optimizer, scheduler, diffusion, train_loader, pbar=None):
        self.args = args
        self.device = device        
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.datalooper = iter(train_loader)
        self.encoder = initialize_encoders(args, device) if args.learn_align else None            
        self.pbar = pbar
        
        self.use_amp = (args.mixed_precision != 'no')
        self.amp_dtype = torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float16
        self.scaler = GradScaler() if args.mixed_precision == 'fp16' else None
    
    def _get_next_batch(self):
        try:
            if self.args.dataset == 'Latent_Pixel':
                images, pixels, labels = next(self.datalooper)
                return images.to(self.device), pixels.to(self.device), labels.to(self.device)
            else:
                images, labels = next(self.datalooper)
                return images.to(self.device), labels.to(self.device) if self.args.class_cond else None
        except StopIteration:
            self.datalooper = iter(self.train_loader)
            return self._get_next_batch()
            
    def _compute_loss(self, images, labels, features):
        model_kwargs = {"y": labels} if self.args.class_cond else {}
        loss_dict = self.diffusion.training_losses(self.model, images, features, model_kwargs=model_kwargs)   
        return loss_dict
    
    def _apply_gradient_clipping(self):
        if self.args.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

    def _update_ema(self):
        if dist_util.is_main_process():
            ema(self.model, self.ema_model, self.args.ema_decay)
                    
    def train_step(self, step):
        self.model.train()
        if self.args.parallel:
            self.train_loader.sampler.set_epoch(step)
        
        grad_accumulation = max(1, self.args.grad_accumulation)
        
        mse_avg = 0.0
        align_avg = 0.0
        total_loss_avg = 0.0

        for accumulation_step in range(grad_accumulation):
            features = None 
            if self.args.dataset == 'Latent_Pixel':
                images, pixels, labels = self._get_next_batch()
                if self.args.learn_align:
                    features = get_feature(self.args, pixels, self.encoder)
            else:
                images, labels = self._get_next_batch()
                if self.args.learn_align:
                    pixels = (images + 1.0) * 127.5
                    features = get_feature(self.args, pixels, self.encoder)
                        
            if self.args.in_chans == 4:
                images = sample_from_latent(images, self.args.latent_scale)  
            
            if (
                self.args.parallel
                and grad_accumulation > 1
                and accumulation_step < grad_accumulation - 1
            ):
                sync_context = self.model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                    loss_dict = self._compute_loss(images, labels, features)
                    loss = loss_dict["loss"].mean() / grad_accumulation
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            total_loss_avg += loss.item()

            if "mse" in loss_dict:
                mse_avg += loss_dict["mse"].mean().item() / grad_accumulation
            
            if self.args.learn_align:
                if "align" in loss_dict:
                    align_avg += loss_dict["align"].mean().item() / grad_accumulation

            if (accumulation_step + 1) % grad_accumulation == 0:
                if self.scaler:
                    if self.args.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                    self._apply_gradient_clipping()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self._apply_gradient_clipping()
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
        
        self.scheduler.step()
        
        if dist_util.is_main_process():
            self._update_ema()
            self.pbar.update(1)
            
            if self.args.learn_align:
                display_stats = OrderedDict([
                    ("align", f"{align_avg:.2f}"),
                    ("mse", f"{mse_avg:.2}")                                  
                ])
                self.pbar.set_postfix(display_stats)
            else:
                self.pbar.set_postfix(mse=f"{mse_avg:.4f}")

        return total_loss_avg
