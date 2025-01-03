import torch
from torch.nn import functional as F
from torch import nn
from typing import Union
import numpy as np
from math import exp
import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision import models

# Structural similarity (SSIM) loss

# SSIM from this repo: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor((_2D_window.expand(channel, 1, window_size, window_size).contiguous()))
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

class SSIMLoss_simple(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss_simple, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = create_window(self.window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        self.window = window
        self.channel = channel
        similarity = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ((1 - similarity))

# Smoothness loss

# PyTorch is NCHW
def gradient_x(img, mode='constant'):
    # use indexing to get horizontal gradients, which chops off one column
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    # pad the results with one zeros column on the right
    return F.pad(gx, (0, 1, 0, 0), mode=mode)

def gradient_y(img, mode='constant'):
    # use indexing to get vertical gradients, which chops off one row
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    # pad the result with one zeros column on bottom
    return F.pad(gy, (0, 0, 0, 1), mode=mode)

def charbonnier_smoothness(flows, alpha=0.3, eps=1e-7):
    return charbonnier(gradient_x(flows), alpha=alpha, eps=eps) + charbonnier(gradient_y(flows), alpha=alpha, eps=eps)

# Reconstruction loss

def charbonnier(tensor, alpha=0.4, eps=1e-4):
    return (tensor * tensor + eps * eps) ** alpha

# L2 regularization, from https://github.com/jbohnslav/deepethogram/blob/ffd7e6bd91f52c7d1dbb166d1fe8793a26c4cb01/deepethogram/losses.py#L61

def should_decay_parameter(name: str, param: torch.Tensor) -> bool:
    if not param.requires_grad:
        return False
    elif 'batchnorm' in name.lower() or 'bn' in name.lower() or 'bias' in name.lower():
        return False
    elif param.ndim == 1:
        return False
    else:
        return True

def get_keys_to_decay(model: nn.Module) -> list:
    to_decay = []
    for name, param in model.named_parameters():
        if should_decay_parameter(name, param):
            to_decay.append(name)
    return to_decay

class L2(nn.Module):
    """L2 regularization
    """
    def __init__(self, model: nn.Module, alpha: float):
        super().__init__()

        self.alpha = alpha
        self.keys = get_keys_to_decay(model)

    def forward(self, model):
        # https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # note that soumith's answer is wrong because it uses W.norm, which takes the square root
        l2_loss = 0 # torch.tensor(0., requires_grad=True)
        for key, param in model.named_parameters():
            if key in self.keys:
                l2_loss += param.pow(2).sum()*0.5

        return l2_loss*self.alpha

# Simple loss

class SimpleLoss(torch.nn.Module):

    def __init__(self,model):
        super(SimpleLoss, self).__init__()
        self.smooth_weights = [.01, .02, .04, .08, .16]
        self.ssim = SSIMLoss_simple(size_average=False)
        self.regularization_criterion = L2(model,1e-5)

    def forward(self, images, reconstructed, flows, model: torch.nn.Module):

    # Find number of scales
        S = len(images)

        # Set weights for flow smoothness loss
        weights = [.01, .02, .04, .08, .16]

        # Calculate loss
        L1s_mean = []
        SSIMs_mean =[]
        smooths_mean = []
        for scale in range(S):
            L1s = charbonnier(images[scale] - reconstructed[scale], alpha=0.4)
            SSIMs = self.ssim(images[scale], reconstructed[scale])
            L1s_mean.append(torch.mean(L1s,dim=[1, 2, 3]))
            SSIMs_mean.append(torch.mean(SSIMs,dim=[1, 2, 3]))
            smooths = charbonnier_smoothness(flows[scale])
            smooths_mean.append(torch.mean(smooths, dim=[1, 2, 3])*weights[scale])

        # Sum across pyramid scales for each loss component!
        SSIM_per_image = torch.stack(SSIMs_mean).sum(dim=0)
        L1_per_image = torch.stack(L1s_mean).sum(dim=0)
        smoothness_per_image = torch.stack(smooths_mean).sum(dim=0)

        # Regularization
        regularization_loss = self.regularization_criterion(model)

        # loss_components = {'reg_loss': regularization_loss.detach(), 
        #                        'SSIM': SSIM_per_image.detach(),
        #                        'L1': L1_per_image.detach(),
        #                        'smoothness': smoothness_per_image.detach(),
        #                        }

        # mean across batch elements
        loss = (torch.mean(SSIM_per_image) + torch.mean(L1_per_image) + torch.mean(smoothness_per_image) + regularization_loss)

        return loss



