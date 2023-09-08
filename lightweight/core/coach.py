import os
# random
import random
from pytorch_lightning.utilities.types import STEP_OUTPUT
# pytorch
import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils import instantiate_from_config, exists
from optimizer import optimizers
from loss import tv
from lpips import LPIPS
from focal_frequency_loss import FocalFrequencyLoss as FFL

class SimpleCoach(pl.LightningModule):
    """
    
    """
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.unet = instantiate_from_config()
        
        # Objectives
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = LPIPS(net=cfg.training.lpips_type)
        self.ff_loss = FFL()
    def training_step(self, batch, batch_idx):
        pred = self.unet(batch['src'])
        gt = batch['tgt']
        loss_dict, loss = self.loss(pred, gt)
        self.log('loss_dict', loss_dict)

        if self.scheduler:
            lr = self.optimizer().param_group[0]['lr']
            self.log('lr', lr)
        return loss


    def loss(self, pred, gt):
        # StudioGAN
        loss = 0.0
        loss_dict = {}
        loss += self.cfg.lmbd_l1 * self.l1_loss(pred, gt)

        if exists(self.cfg.lmbd_lpips) and self.current_epoch >= self.lpips_start:
            loss += self.cfg.lmbd_lpips * self.lpip_loss(gt, pred)
        if exists(self.cfg.lmbd_lpips) and self.current_epoch >= self.lpips_start:
            loss += self.cfg.lmbd_lpips * self

        return loss_dict, loss

    def log(self,):
        ...

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        ...


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        # keyword 없는거 무시할 수 있는 코드
        opt = optimizers[self.opt](params, lr=lr, )
        return opt

class GANCoach(SimpleCoach):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        
class Distiller(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        
    def training_step(self, batch):
        ...

    def loss(self):
        ...
    @torch.no_grad()
    def make_sample(self, batch):
        if self.cfg.gan_teacher is True:
            ...