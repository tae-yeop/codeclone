import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k)

    def forward(self, q, k, v, is_cross, place_in_unet, num_heads, **kwargs):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class MutualSelfAttentionControl(AttentionBase):

    def __init__(self, start_step=4, start_layer=10,
                 layer_idx=None, step_idx=None, total_step=50, guidance_scale=7.5):
        