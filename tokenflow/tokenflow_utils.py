from typing import Type
import torch
import os


from util import isinstance_str, batch_cosine_sim

def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)


def register_batchidx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0,1], 1:[0, 1], 2:[0,1]}
    up_res_dict = {1: [0,1,2], 2: [0,1,2], 3:[0,1,2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attnetions[block].trasnformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents

def register_conv_injection(model, injection_schedule):
    def conv_forward(self,):
        def forward(input_tensor, temb):
            hidden_states = input_tensor
            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.co

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
        hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timesteps)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        norm_hidden_states = norm_hidden_states.view(3 n_frames, seqeunce_length, dim)
        if self.pivotal_pass:
            self.pivot_hidden_states = norm_hidden_states
        else:
            idx1 = []
            idx2 = []
            batch_idxs = [self.batch_idx]
            if self.batch_idx > 0:
                batch_idxs.append(self.batch_idx -1)

            sim = batch_cosine_sim(norm_hidden_states[0].reshape(-1, dim),
                                   self.pivot_)

            if len(batch_idxs) == 2:
                sim1, sim2 = sim.chunck(2, dim=1)
                idx1.append(sim1.argmax(dim=-1))
                idx2.append(sim2.argmax(dim=-1))
            else:
                idx1.append(sim.argmax(dim=-1))
            idx1 = torch.stack(idx1 * 3, dim=0)
            idx1 = idx1.squeeze(1)

        # SA
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if self.pivotal_pass:
            if len(batch_idxs) == 2:
                attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                attn_output1

                s = torch.arange(0, n_frames).to(idx1.device) + batch_idxs[0] * n_frames

                p1 = batch_idxs[0] * n_frames

    return TokenFlowBlock

# TransformerBlk
def set_tokenflow(model: torch.nn.Module):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tokenflow_block_fn = make_tokenflow_attention_block
            module.__classs__ = make_tokenflow_block_fn(module.__class__)

            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False
        
    return model