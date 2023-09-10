from typing import Type
import torch
import os


from util import isinstance_str, batch_cosine_sim


def make_tokenflow_attention_block(block_class):

    class TokenFlowBlock(block_class):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_lables=None
        ):
        batch_size, sequence_length, dim = hidden_states.shape
        n_frames = batch_size // 3
        mid_idx = n_frames // 2
        hidden_states = 

    return TokenFlowBlock

def set_tokenflow(model: torch.nn.Module):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tokenflow_block_fn = make_tokenflow_attention_block
            module.__classs__ = make_tokenflow_block_fn(module.__class__)

            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False
        
    return model