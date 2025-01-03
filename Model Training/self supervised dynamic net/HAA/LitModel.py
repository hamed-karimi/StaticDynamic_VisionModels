import lightning.pytorch as pl
import torch.nn as nn
import torch
import sys
import gc
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR
import torchvision.transforms as transforms
# import torch.utils.model_zoo as model_zoo
import feafa_architecture
import feafa_criterion
import feafa_utils
import psutil


class LitModel(pl.LightningModule):
    def __init__(self, lr, momentum, weight_decay, lr_steps=[100, 200]):
        super().__init__()
        self.encoder = feafa_architecture.TinyMotionNet()
        self.decoder = feafa_utils.Reconstructor()
        self.criterion = feafa_criterion.SimpleLoss(self.encoder)
        
        self.register_buffer('lr', lr)
        self.register_buffer('weight_decay', weight_decay)
        self.register_buffer('momentum', momentum)
        self.register_buffer('lr_steps', torch.tensor(lr_steps))


    def training_step(self, batch, batch_idx):
        __key__, frames = batch
        if batch_idx % 100 == 0:
            print('cpu memory usage 1: ', psutil.virtual_memory().used / 1000000000)
            # gc.collect()
        flows = self.encoder(frames)
        t0s, reconstructed, flows_reshaped = self.decoder(frames, flows) # t0s are original images excluding 
                                                                               # the 11th, downsampled to match the 
                                                                               # reconstructed versions
   
        loss = self.criterion(t0s, reconstructed, flows_reshaped, self.encoder)
        return loss

    def validation_step(self, batch, batch_idx):
        __key__, frames = batch
        flows = self.encoder(frames)
        t0s, reconstructed, flows_reshaped = self.decoder(frames, flows)
        loss = self.criterion(t0s, reconstructed, flows_reshaped, self.encoder)
        self.log('val_loss', loss, sync_dist=True)

    def validation_step_end(self, batch_parts):
        print('validation parts: ', batch_parts.shape)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.encoder.parameters(), 
                                    self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "interval": "epoch",
                "scheduler": MultiplicativeLR(optimizer, 
                                              lr_lambda=lambda x: 0.1 ** (sum(self.current_epoch >= self.lr_steps))),
                "frequency": 1
            },
        }