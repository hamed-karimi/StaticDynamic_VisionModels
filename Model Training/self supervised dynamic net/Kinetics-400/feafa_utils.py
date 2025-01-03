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


class Normalizer:
    """Allows for easy z-scoring of tensors on the GPU.
    Example usage: You have a tensor of images of shape [N, C, H, W] or [N, T, C, H, W] in range [0,1]. You want to
        z-score this tensor.
    Methods:
        process_inputs: converts input mean, std into a torch tensor
        no_conversion: dummy method if you don't actually want to standardize the data
        handle_tensor: deals with self.mean and self.std depending on inputs. Example: your Tensor arrives on the GPU
            but your self.mean and self.std are still on the CPU. This method will move it appropriately.
        denormalize: converts standardized arrays back to their original range
        __call__: z-scores input data
    Instance variables:
        mean: mean of input data. For images, should have 2 or 3 channels
        std: standard deviation of input data
    """
    def __init__(self,
                 mean: Union[list, np.ndarray, torch.Tensor] = None,
                 std: Union[list, np.ndarray, torch.Tensor] = None,
                 clamp: bool = True):
        """Constructor for Normalizer class.
        Args:
            mean: mean of input data. Should have 3 channels (for R,G,B) or 2 (for X,Y) in the optical flow case
            std: standard deviation of input data.
            clamp: if True, clips the output of a denormalized Tensor to between 0 and 1 (for images)
        """
        # make sure that if you have a mean, you also have a std
        # XOR
        has_mean, has_std = mean is None, std is None
        assert (not has_mean ^ has_std)

        self.mean = self.process_inputs(mean)
        self.std = self.process_inputs(std)
        # prevent divide by zero, but only change values if it's close to 0 already
        if self.std is not None:
            assert (self.std.min() > 0)
            self.std[self.std < 1e-8] += 1e-8
        self.clamp = clamp

    def process_inputs(self, inputs: Union[torch.Tensor, np.ndarray]):
        """Deals with input mean and std.
        Converts to tensor if necessary. Reshapes to [length, 1, 1] for pytorch broadcasting.
        """
        if inputs is None:
            return (inputs)
        if type(inputs) == list: # or type(inputs) == int:
            inputs = np.array(inputs).astype(np.float32)
        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs)
        #print(inputs)
        assert (type(inputs) == torch.Tensor)
        inputs = inputs.float()
        C = inputs.shape[0]
        inputs = inputs.reshape(C, 1, 1)
        inputs.requires_grad = False
        return inputs

    def no_conversion(self, inputs):
        """Dummy function. Allows for normalizer to be called when you don't actually want to normalize.
        That way we can leave normalize in the training loop and only optionally call it.
        """
        return inputs

    def handle_tensor(self, tensor: torch.Tensor):
        """Reshapes std and mean to deal with the dimensions of the input tensor.
        Args:
            tensor: PyTorch tensor of shapes NCHW or NCTHW, depending on if your CNN is 2D or 3D
        Moves mean and std to the tensor's device if necessary
        If you've stacked the C dimension to have multiple images, e.g. 10 optic flows stacked has dim C=20,
            repeats self.mean and self.std to match
        """
        if tensor.ndim == 4:
            N, C, H, W = tensor.shape
        elif tensor.ndim == 5:
            N, C, T, H, W = tensor.shape
        else:
            raise ValueError('Tensor input to normalizer of unknown shape: {}'.format(tensor.shape))

        t_d = tensor.device
        if t_d != self.mean.device:
            self.mean = self.mean.to(t_d)
        if t_d != self.std.device:
            self.std = self.std.to(t_d)

        c = self.mean.shape[0]
        if c < C:
            # handles the case where instead of N, C, T, H, W inputs, we have concatenated
            # multiple images along the channel dimension, so it's
            # N, C*T, H, W
            # this code simply repeats the mean T times, so it's
            # [R_mean, G_mean, B_mean, R_mean, G_mean, ... etc]
            n_repeats = C / c
            assert (int(n_repeats) == n_repeats)
            n_repeats = int(n_repeats)
            repeats = tuple([n_repeats] + [1 for i in range(self.mean.ndim - 1)])
            self.mean = self.mean.repeat((repeats))
            self.std = self.std.repeat((repeats))

        if tensor.ndim - self.mean.ndim > 1:
            # handles the case where our inputs are NCTHW
            self.mean = self.mean.unsqueeze(-1)
        if tensor.ndim - self.std.ndim > 1:
            self.std = self.std.unsqueeze(-1)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converts normalized data back into its original distribution.
        If self.clamp: limits output tensor to the range (0,1). For images
        """
        if self.mean is None:
            return tensor

        # handles dealing with unexpected shape of inputs, wrong devices, etc.
        self.handle_tensor(tensor)
        tensor = (tensor * self.std) + self.mean
        if self.clamp:
            tensor = tensor.clamp(min=0.0, max=1.0)
        return tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes input data"""
        if self.mean is None:
            return tensor

        # handles dealing with unexpected shape of inputs, wrong devices, etc.
        self.handle_tensor(tensor)

        tensor = (tensor - self.mean) / (self.std)
        return tensor

class Resample2d(torch.nn.Module):
    """ Module to sample tensors using Spatial Transformer Networks. Caches multiple grids in GPU VRAM for speed.
    Examples
    -------
    model = MyOpticFlowNetwork()
    resampler = Resample2d(device='cuda:0')
    # flows map t0 -> t1
    flows = model(images)
    t0 = resampler(images, flows)
    model = MyStereoNetwork()
    resampler = Resample2d(device='cuda:0', horiz_only=True)
    disparity = model(left_images, right_images)
    """

    def __init__(self, size: Union[tuple, list] = None, device: Union[str, torch.device] = None,
                 horiz_only: bool = False,
                 num_grids: int = 5):
        """ Constructor for resampler.
        Parameters
        ----------
        size: tuple, list. shape: (2,)
            height and width of input tensors to sample
        device: str, torch.device
            device on which to store affine matrices and grids
        horiz_only: bool
            if True, only resample in the X dimension. Suitable for stereo matching problems
        num_grids: int
            Number of grids of different sizes to cache in GPU memory.
            A "Grid" is a tensor of shape H, W, 2, with (-1, -1) in the top-left to (1, 1) in the bottom-right. This
            is the location at which we will sample the input images. If we don't do anything to this grid, we will
            just return the original image. If we add our optic flow, we will sample the input image at at the locations
            specified by the optic flow.
            In many flow and stereo matching networks, images are warped at multiple resolutions, e.g.
            1/2, 1/4, 1/8, and 1/16 of the original resolution. To avoid making a new grid for sampling 4 times every
            time this is called, we will keep these 4 grids in memory. If there are more than `num_grids` resolutions,
            it will calculate and discard the top `num_grids` most frequently used grids.
        """
        super().__init__()
        if size is not None:
            assert (type(size) == tuple or type(size) == list)
        self.size = size

        # identity matrix
        # self.base_mat = torch.Tensor([[1, 0, 0], [0, 1, 0]])
        self.device = device
        self.horiz_only = horiz_only
        self.num_grids = num_grids
        self.sizes = []
        self.grids = []
        self.uses = []

    def forward(self, images: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """ resample `images` according to `flow`
        Parameters
        ----------
        images: torch.Tensor. shape (N, C, H, W)
            images
        flow: torch.Tensor. shape (N, 2, H, W) in the flow case or (N, 1, H, W) in the stereo matching case
            should be in ABSOLUTE PIXEL COORDINATES.
        Returns
        -------
        resampled: torch.Tensor (N, C, H, W)
            images sampled at their original locations PLUS the input flow
        """
        # for MPI-sintel, assumes t0 = Resample2d()(t1, flow)
        self.device = flow.device
        self.base_mat = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=self.device)
        if self.size is not None:
            H, W = self.size
        else:
            H, W = flow.size(2), flow.size(3)
        # print(H,W)
        # images: NxCxHxW
        # flow: Bx2xHxW
        grid_size = [flow.size(0), 2, flow.size(2), flow.size(3)]
        if not hasattr(self, 'grids') or grid_size not in self.sizes:
            if len(self.sizes) >= self.num_grids:
                min_uses = min(self.uses)
                min_loc = self.uses.index(min_uses)
                del (self.uses[min_loc], self.grids[min_loc], self.sizes[min_loc])
            # make the affine mat match the batch size
            self.affine_mat = self.base_mat.repeat(images.size(0), 1, 1)

            # function outputs N,H,W,2. Permuted to N,2,H,W to match flow
            # 0-th channel is x sample locations, -1 in left column, 1 in right column
            # 1-th channel is y sample locations, -1 in first row, 1 in bottom row
            # this_grid = F.affine_grid(self.affine_mat, images.shape, align_corners=False).permute(0, 3, 1, 2).to(
            #     self.device)
            this_grid = F.affine_grid(self.affine_mat, images.shape, align_corners=False).permute(0, 3, 1, 2)
            this_size = [i for i in this_grid.size()]
            self.sizes.append(this_size)
            self.grids.append(this_grid)
            self.uses.append(0)
            # print(this_grid.shape)
        else:
            grid_loc = self.sizes.index(grid_size)
            this_grid = self.grids[grid_loc]
            self.uses[grid_loc] += 1

        # normalize flow
        # input should be in absolute pixel coordinates
        # this normalizes it so that a value of 2 would move a pixel all the way across the width or height
        # horiz_only: for stereo matching, Y values are always the same
        if self.horiz_only:
            # flow = flow[:, 0:1, :, :] / ((W - 1.0) / 2.0)
            flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                              torch.zeros((flow.size(0), flow.size(1), H, W))], 1)
        else:
            # for optic flow matching: can be displaced in X or Y
            flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                              flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        # sample according to grid + flow
        # print('this_grid device: ', this_grid.device)
        # print('flow device:', flow.device)
        return F.grid_sample(input=images, grid=(this_grid + flow).permute(0, 2, 3, 1),
                             mode='bilinear', padding_mode='border', align_corners=False)

class Reconstructor:
    def __init__(self):
        # device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        # print("device in reconstructor is ", device)
        self.resampler = Resample2d()
        mean = [0.5,0.5,0.5] # from here https://github.com/jbohnslav/deepethogram/blob/ffd7e6bd91f52c7d1dbb166d1fe8793a26c4cb01/deepethogram/conf/augs.yaml 
        std = [0.5,0.5,0.5]
        self.normalizer = Normalizer(mean=mean, std=std)

    def reconstruct_images(self, image_batch: torch.Tensor, flows: Union[tuple, list]):
        # SSIM DOES NOT WORK WITH Z-SCORED IMAGES
        # requires images in the range [0,1]. So we have to denormalize for it to work!
        image_batch = self.normalizer.denormalize(image_batch)
        if image_batch.ndim == 4:
            N, C, H, W = image_batch.shape
            num_images = int(C / 3) - 1
            t0 = image_batch[:, :num_images * 3, ...].contiguous().view(N * num_images, 3, H, W)
            t1 = image_batch[:, 3:, ...].contiguous().view(N * num_images, 3, H, W)
        elif image_batch.ndim == 5:
            N, C, T, H, W = image_batch.shape
            num_images = T - 1
            t0 = image_batch[:, :, :num_images, ...]
            t0 = t0.transpose(1, 2).reshape(N * num_images, C, H, W)
            t1 = image_batch[:, :, 1:, ...]
            t1 = t1.transpose(1, 2).reshape(N * num_images, C, H, W)
        else:
            raise ValueError('unexpected batch shape: {}'.format(image_batch))

        reconstructed = []
        t1s = []
        t0s = []
        flows_reshaped = []
        for flow in flows:
            # upsampled_flow = F.interpolate(flow, (h,w), mode='bilinear', align_corners=False)
            if flow.ndim == 4:
                n, c, h, w = flow.size()
                flow = flow.view(N * num_images, 2, h, w)
            else:
                n, c, t, h, w = flow.shape
                flow = flow.transpose(1, 2).reshape(n * t, c, h, w)

            downsampled_t1 = F.interpolate(t1, (h, w), mode='bilinear', align_corners=False)
            downsampled_t0 = F.interpolate(t0, (h, w), mode='bilinear', align_corners=False)
            t0s.append(downsampled_t0)
            t1s.append(downsampled_t1)
            reconstructed.append(self.resampler(downsampled_t1, flow))
            del (downsampled_t1, downsampled_t0)
            flows_reshaped.append(flow)

        return tuple(t0s), tuple(reconstructed), tuple(flows_reshaped)

    def __call__(self, image_batch: torch.Tensor, flows: Union[tuple, list]):
        return self.reconstruct_images(image_batch, flows)

