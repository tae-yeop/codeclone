from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalControlnetMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from diffusers.models.unet_2d_condition import UNet2DConditionModel



def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    def __init__(self, 
                 conditioning_embedding_channels: int, # latent와 합쳐질 수 있게 같은 channles 수
                 conditioning_channels: int = 3,
                 block_out_channels: Tuple[int] = (16, 32, 96, 256),):
        super().__init__()

        # RGB 3차원을 최종적으로 256까지 채널을 늘리면서 (512, 512) -> (64, 64)
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class ControlNetModel(ModelMixin, ConfigMixin, FromOriginalControlnetMixin):

    @register_to_config
    def __init__():
        super().__init__()
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)


    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
    ):
        ...


    def forward(
        self,
        sample,
        timestep,):

        



        