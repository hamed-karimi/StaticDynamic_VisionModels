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

# Building blocks

def conv(batchNorm: bool, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, bias: bool = True):
    """ Convenience function for conv2d + optional BN + leakyRELU """
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )

def crop_like(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Crops input to target's H,W  """
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def deconv(in_planes: int, out_planes: int, bias: bool = True):
    """ Convenience function for ConvTranspose2d + leakyRELU """
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )

class Interpolate(nn.Module):
    """ Wrapper to be able to perform interpolation in a nn.Sequential
    Modified from the PyTorch Forums:
    https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588/2
    """

    def __init__(self, size=None, scale_factor=None, mode: str = 'bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area']
        self.mode = mode
        if self.mode == 'nearest':
            self.align_corners = None
        else:
            self.align_corners = False

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x

def i_conv(batchNorm: bool, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, bias: bool = True):
    """ Convenience function for conv2d + optional BN + no activation """
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=bias),
        )


def predict_flow(in_planes: int, out_planes: int = 2, bias: bool = False):
    """ Convenience function for 3x3 conv2d with same padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)


def get_hw(tensor):
    """ Convenience function for getting the size of the last two dimensions in a tensor """
    return tensor.size(-2), tensor.size(-1)


class CropConcat(nn.Module):
    """ Module for concatenating 2 tensors of slightly different shape.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, tensors: tuple) -> torch.Tensor:
        assert type(tensors) == tuple
        hs, ws = [tensor.size(-2) for tensor in tensors], [tensor.size(-1) for tensor in tensors]
        h, w = min(hs), min(ws)

        return torch.cat(tuple([tensor[..., :h, :w] for tensor in tensors]), dim=self.dim)


def conv3d(in_planes: int, out_planes: int, kernel_size: Union[int, tuple] = 3, stride: Union[int, tuple] = 1,
           bias: bool = True, batchnorm: bool = True, act: bool = True, padding: tuple = None):
    """ 3D convolution
    Expects inputs of shape N, C, D/F/T, H, W.
    D/F/T is frames, depth, time-- the extra axis compared to 2D convolution.
    Returns output of shape N, C_out, D/F/T_out, H_out, W_out.
    Out shape will be determined by input parameters. for more information see PyTorch docs
    https://pytorch.org/docs/master/generated/torch.nn.Conv3d.html
    Args:
        in_planes: int
            Number of channels in input tensor.
        out_planes: int
            Number of channels in output tensor
        kernel_size: int, tuple
            Size of 3D convolutional kernel. in order of (D/F/T, H, W). If int, size is repeated 3X
        stride: int, tuple
            Stride of convolutional kernel in D/F/T, H, W order
        bias: bool
            if True, adds a bias parameter
        batchnorm: bool
            if True, adds batchnorm 3D
        act: bool
            if True, adds LeakyRelu after (optional) batchnorm
        padding: int, tuple
            padding in T, H, W. If int, repeats 3X. if none, "same" padding, so that the inputs are the same shape
            as the outputs (assuming stride 1)
    Returns:
        nn.Sequential with conv3d, (batchnorm), (activation function)
    """
    modules = []
    if padding is None and type(kernel_size) == int:
        padding = (kernel_size - 1) // 2
    elif padding is None and type(kernel_size) == tuple:
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
    else:
        raise ValueError('Unknown padding type {} and kernel_size type: {}'.format(padding, kernel_size))

    modules.append(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias))
    if batchnorm:
        modules.append(nn.BatchNorm3d(out_planes))
    if act:
        modules.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*modules)


def deconv3d(in_planes: int, out_planes: int, kernel_size: int = 4, stride: int = 2, bias: bool = True,
             batchnorm: bool = True, act: bool = True, padding: int = 1):
    """ Convenience function for ConvTranspose3D. Optionally adds batchnorm3d, leakyrelu """
    modules = [nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                  bias=bias, padding=padding)]
    if batchnorm:
        modules.append(nn.BatchNorm3d(out_planes))
    if act:
        modules.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*modules)


def predict_flow_3d(in_planes: int, out_planes: int):
    """ Convenience function for conv3d, 3x3, no activation or batchnorm """
    return conv3d(in_planes, out_planes, kernel_size=3, stride=1, bias=True, act=False, batchnorm=False)

# Network class

class TinyMotionNet(nn.Module):

    def __init__(self,num_images=11, batchNorm=True):
        super().__init__()
        self.num_images = num_images
        self.input_channels = self.num_images * 3
        self.output_channels = int((num_images - 1) * 2)

        self.batchNorm = batchNorm
        self.flow_div = 1

        self.conv1 = conv(self.batchNorm, self.input_channels, 64, kernel_size=7)
        self.conv2 = conv(self.batchNorm, 64, 128, stride=2, kernel_size=5)
        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv4 = conv(self.batchNorm, 256, 128, stride=2)

        self.deconv3 = deconv(128, 128)
        self.deconv2 = deconv(128, 64)

        self.xconv3 = i_conv(self.batchNorm, 384 + self.output_channels, 128)
        self.xconv2 = i_conv(self.batchNorm, 192 + self.output_channels, 64)

        self.predict_flow4 = predict_flow(128, out_planes=self.output_channels)
        self.predict_flow3 = predict_flow(128, out_planes=self.output_channels)
        self.predict_flow2 = predict_flow(64, out_planes=self.output_channels)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 4, 2, 1)

        self.concat = CropConcat(dim=1)
        self.interpolate = Interpolate

    def forward(self,x):
        # print('TinyMotionNet device: ', x.device)
        N, C, T, H, W = x.shape
        x = x.view(N,C*T,H,W)
        out_conv1 = self.conv1(x)  # 1 -> 1
        out_conv2 = self.conv2(out_conv1)  # 1 -> 1/2
        out_conv3 = self.conv3(out_conv2)  # 1/2 -> 1/4
        out_conv4 = self.conv4(out_conv3)  # 1/4 -> 1/8

        flow4 = self.predict_flow4(out_conv4) * self.flow_div
        # see motionnet.py for explanation of multiplying by 2
        flow4_up = self.upsampled_flow4_to_3(flow4) * 2
        out_deconv3 = self.deconv3(out_conv4)

        concat3 = self.concat((out_conv3, out_deconv3, flow4_up))
        out_interconv3 = self.xconv3(concat3)
        flow3 = self.predict_flow3(out_interconv3) * self.flow_div
        flow3_up = self.upsampled_flow3_to_2(flow3) * 2
        out_deconv2 = self.deconv2(out_interconv3)

        concat2 = self.concat((out_conv2, out_deconv2, flow3_up))
        out_interconv2 = self.xconv2(concat2)
        flow2 = self.predict_flow2(out_interconv2) * self.flow_div

        flow1 = F.interpolate(flow2, (H, W), mode='bilinear', align_corners=False) * 2
        # flow2*=self.flow_div
        # flow3*=self.flow_div
        # flow4*=self.flow_div
        # import pdb
        # pdb.set_trace()

        # if self.training:
        #     return flow1, flow2, flow3, flow4
        # else:
        #     return flow1,
        return flow1, flow2, flow3, flow4



