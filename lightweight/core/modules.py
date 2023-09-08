import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_wavelets import DWTInverse, DWTForward

from math import floor, ceil
from einops import rearrange
from typing import Optional, List, Tuple


def conv_bn(in_channels, out_channels, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv3x3', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                        kernel_size=3, stride=stride, padding=padding, groups=groups,bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('conv1x1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                        kernel_size=1, stride=stride, padding=padding, groups=groups,bias=False))

    return result


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, kernel=3, stride=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2 if stride is None else stride
        self.op = nn.Conv2d(channels, self.out_channels, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.op(x)
        return x
    

class Upsample(nn.Module):
    def __init__(self, channels, out_channels=None, paddding=1): 
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=1, padding=1)
         
    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.op(x)
        return x

class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 inference_mode=False, stride=1, padding=0, dilation=1,
                 groups=1, num_conv_branches=1, zero_params=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inference_mode = inference_mode
        self.stride = stride
        self.groups = groups
        self.num_conv_branches = num_conv_branches

        
        self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

            if zero_params:
                zero_module(self.rbr_skip) if self.rbr_skip is not None else ...
                zero_module(self.rbr_conv) 
                zero_module(self.rbr_scale) if self.rbr_scale is not None else ...
    def forward(self, x):
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))
    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list
    
    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    
class CostomAdaptiveAvgPool2D(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x):
        H_in,  W_in  = x.shape[2:]
        H_out, W_out = [self.output_size, self.output_size] \
                       if isinstance(self.output_size, int) \
                       else self.output_size
        
        out_i = []
        for i in range(H_out):
            out_j = []
            for j in range(W_out):
                
                hs = int(floor(i * H_in / H_out))
                he = int(ceil((i+1) * H_in / H_out))
                
                ws = int(floor(j * W_in / W_out))
                we = int(ceil((j+1) * W_in / W_out))
                
                # print(hs, he, ws, we)
                kernel_size = [he-hs, we-ws]
                
                out = F.avg_pool2d(x[:, :, hs:he, ws:we], kernel_size) 
                out_j.append(out)
            out_j = torch.concat(out_j, -1)
            out_i.append(out_j)
        out_i = torch.concat(out_i, -2)
        return out_i
    

class WaveletGating(nn.Module):
    def __init__(self, channels, pool_size=4):
        super().__init__()
        self.pool_size = pool_size
        # self.avgpool = nn.AvgPool2d(self.pool_size)
        self.avgpool = CostomAdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, channels //2, kernel_size=1, bias=False),
                                nn.ReLU(True),
                                nn.Conv2d(channels //2, 4, kernel_size=1, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class WGDown(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='haar')
        self.wavelet_gating = WaveletGating(channels)
        
    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq
    
    def forward(self, x):
        """
        Args:
            x : [B, C_in, H, W]
        Returns:
            [B, C_in, H//2, W//2]
        """
        # dwt를 하고 나오는 순서 : LL, LH, HL, HH
        # LL, high = self.dwt(x)
        # LH, HL, HH = tuple(rearrange(high[0] , 'b c n h w -> n b c h w ', n=3))
        # b, _, _, h, w = high[0].size()
        # [1, 4, 1, 1]
        # low, high = self.dwt
        # score = self.wavelet_gating(x)
        # [B, C_in, 4, H//2, W//2]
        # result = torch.concat((low.unsqueeze(2), high[0]), dim=2) * score.unsqueeze(1)
        # return result.flatten(start_dim=1, end_dim=2)
        LL, high = self.dwt(x)
        # LH, HL, HH = tuple(rearrange(high[0] , 'b c n h w -> n b c h w ', n=3))
        score = self.wavelet_gating(x)
        # LL = LL * score[:, 0, ...]
        # LH = LH * score[:, 1, ...]
        # HL = HL * score[:, 2, ...]
        # HH = HH * score[:, 3, ...]
        # result = LL + LH + HL + HH
        # Stack along a new dimension
        # tensors = torch.stack([LL, LH, HL, HH], dim=1) # tensors shape: [B, 4, C_in, H//2, W//2]
        # print(low.unsqueeze(2).shape)
        # print(high[0])
        result = torch.concat((LL.unsqueeze(2), high[0]), dim=2) * score.unsqueeze(1)
        result = result.sum(dim=1)
        return result

class WGUp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.idwt = DWTInverse(mode="zero", wave='haar')
        self.wavelet_gating = WaveletGating(channels)

    def dwt_to_img(img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return op((low, [high]))
    def forward(self, x):
        """
        Args:
            x : [B, C_in, H, W]
        Returns:
            [B, C_in, 2*H, 2*W]
        """
        score = self.wavelet_gating(x) # [B, 4, 1, 1]
        # x = x.chunk(4, dim=1)
        b, c, h, w = x.size()
        x_reshaped = x.view(b, 4, c//4, h, w)
        # score_reshaped = score.expand(b, 4, c//4, h, w)
        freq = x_reshaped * score.unsqueeze(2)
        
        result = self.idwt((freq[:,0,...], [freq[:, 1:, ...].flatten(start_dim=1, end_dim=2)]))
        return result