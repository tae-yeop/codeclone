from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin



from .embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


@dataclass
class UNet2DOutput(BaseOutput):
    sample: torch.FloatTensor

class UNet2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__():
        super().__init__()



    def forward(self, sample,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states,
                down_block_additional_residuals = None, # adapter 혹은 controlnet에서 오는 feature
                mid_block_additional_residual = None, # adapter 혹은 controlnet에서 오는 feature
                **kwargs
                ):

        ...
        # 0. 설정
        # up block에서 upsample 해서 해상도를 키워야 하는 단계가 있다
        forward_upsample_size = False

        # 만약에 UNet으로 들어가는 인풋 (b, 4, 64, 64)의 해상도 64, 64만큼 나중에 upblock에서 만들어내야한다
        # 그런데 현재 UNet의 upblock에 있는 upsampler 갯수로 불충분한 경우 64, 64로 다시 못 키우는 경우를 대비해서
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        
        # 1. time
        timesteps = timestep
        ...
        # time embedding을 얻음
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb, timestep_cond)


        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        # controlnet 혹은 adapter 쪽에서 오는 인풋이 있는지 체크
        # controlnet의 mid block과 down block 쪽에서 나온 feature가 있는지 체크
        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # adapter의 mid block은 없으나 down block 쪽에서 나오는 feature가 있는지 체크
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            # 보통 CA를 있기 때문에 여길 실행
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # 만약 T2I adapter를 사용하는 상황이면 down_block_additional_residuals에 있는 첫번째 값을 down block에 전달
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)
                    
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                ...

            down_block_res_samples += res_samples
        # 만약 controlnet을 쓰는거라면
        # 방금 down block에 나온 residual feature를 contronlet에서 나온 feature와 더해줌 => up block에 전달하기 위해
        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                # 더한 뒤에 다시 튜플로 넣어줌
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            # upblock에 skip connection할 feature완성
            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb,
                                    encoder_hidden_states=encoder_hidden_states,
                                    attention_mask=attention_mask,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    encoder_attention_mask=encoder_attention_mask,
                                )

        # controlnet에서 오는 feature가 잇으면 합쳐줌
        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            # Up block에 들어갈 skip connection 설정
            # 먼저 제일 뒤에서부터 resdiual feature 뽑아냄
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            # 뽑아낸 제일 마지막은 제외해서 리스트를 다시 구성 (loop 돌면서 뒤에서 부터 하나씩 pop)
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # final block이 아닌 경우는 upsample_block에서 upsample할 수 있게
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

                
            # CA있으므로 이거 사용
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                ...

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        # 마지막 VAE 디코더로 들어가게끔 채널수 맞춰줌
        sample = self.conv_out(sample)

        ...
        return UNet2DConditionOutput(sample=sample)