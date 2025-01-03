import lightning.pytorch as pl
import torch.nn as nn
import torch
import sys
import gc
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR
import torchvision.transforms as transforms
from rgb_resnet import rgb_resnet18
import psutil

class LitModel(pl.LightningModule):
    def __init__(self, lr, momentum, weight_decay, lr_steps=[100, 200]):
        super().__init__()
        self.model = rgb_resnet18(pretrained=True, num_classes=500)
        self.criterion = nn.CrossEntropyLoss()
        self.register_buffer('lr', lr)
        self.register_buffer('weight_decay', weight_decay)
        self.register_buffer('momentum', momentum)
        self.register_buffer('lr_steps', torch.tensor(lr_steps))
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if batch_idx % 25 == 0:
            print('cpu memory usage 1: ', psutil.virtual_memory().used / 1000000000)
        image, label = batch[0], batch[1]
        y_hat = self.model(image)
        loss = self.criterion(y_hat, label)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch[0], batch[1]
        y_hat = self.model(image)
        top3_labels = y_hat.topk(k=3, dim=1).indices

        loss = self.criterion(y_hat, label)
        pred_acc = torch.zeros_like(top3_labels)
        for d in range(top3_labels.shape[1]):
            pred_acc[:, d] = (label == top3_labels[:, d])

        top1_acc = pred_acc[:, 0]
        top3_acc = pred_acc.sum(dim=1)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_top1_acc', top1_acc.float().mean(), sync_dist=True)
        self.log('val_top3_acc', top3_acc.float().mean(), sync_dist=True)

    def validation_step_end(self, batch_parts):
        print('validation parts: ', batch_parts.shape)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), 
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