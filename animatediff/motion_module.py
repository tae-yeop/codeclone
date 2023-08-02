from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.utils import BaseOutput

from einops import rearrange, repeat



@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor

def get_motion_module(
    in_channels,
    motion_module_type:str,
    motion_module_kwargs:dict
):
    if motion_module_type == 'Vanilla':
        return VanillaTemporalModule(in_channels=in_channels,
                                     **motion_module_kwargs)
    else:
        raise ValueError

    
class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads = 8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len = 24,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)
        output = hidden_states
        return output

# 
class TemporalTransformer3DModel(nn.Module):
    def __init__(self,
                 in_channels,
                 num_attention_heads,
                 attention_head_dim,
                 ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        