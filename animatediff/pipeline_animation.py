# https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py
import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

# 아웃풋 정의
from diffusers.utils import is_accelerate_available, BaseOutput

# SDPL이 아닌 더 상위 파이프라인
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    # SDPL이면 ["safety_checker", "feature_extractor"] => init에도 포함됨
    _optional_components = [] 

    def __init__(self,
                 vae,
                 text_encoder,
                 tokenizer,
                 unet,
                 scheduler):
        super.__init__()

        # SDPL와 동일, 차이점은 safety_checker, feature_extractor 없음
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)


        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)


        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        # setattr 하는 부분, SDPL과 동일
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()
 
    # 없는거 : enable_vae_tiling, disable_vae_tiling, enable_model_cpu_offload

    # DiffusinPipelne과 조금 다르다
    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)
    # DiffusinPipelne과 조금 다르다
    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device



    @torch.no_grad()
    def __call__(self,
                 prompt:Union[str, List[str]],
                 video_length:Optional[int],
                 height:Optional[int]=None,
                 width:Optional[int]=None,
                 num_inference_steps:int=50,
                 guidance_scale:float=7.5,
                 negative_prompt:Optional[Union[str, List[str]]]=None,
                 num_videos_per_prompt:Optional[int]=1, # <- 추가
                 eta:float=0.0,
                 generator:Optional[Union[torch.Generator, List[torch.Generator]]]=None,
                 latents:Optional[torch.FloatTensor]=None,
                 output_type:Optional[str]='tensor',
                 return_dict:bool=Ture,
                 callback:Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps:Optional[int]=1,
                 **kwrags):


        # 0. Default height and width to unet
        # scale_factor가 사이즈 조절 factor인듯
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)


        # 2. Define call parameters
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        # negative 여러번
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
        # lora scale은 반영하지 않는다 : text_encoder_lora_scale
        text_embeddings = self._encode_prompt(prompt, 
                                              device, 
                                              num_videos_per_prompt, 
                                              do_classifier_free_guidance, 
                                              negative_prompt)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.uet.in_channels
        latents = self.prepare_latents(batch_size * num_videos_per_prompt,
                                       num_channels_latents,
                                       video_length, # 추가됨
                                       height,
                                       width,
                                       text_embeddings.dtype,
                                       device,
                                       generator,
                                       latents)
        # latent type을 따로 저장함
        latents_dtype = latents.dtype

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                # 여기서 다른점 : 아웃풋에서 바로 sample을 얻음
                # [B, C, F, H, W]
                noise_pred = self.unet(latent_model_input,
                                       t,
                                       encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)

                # perform guidance : SDPL과 동일
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # SDPL에 추가적인 사항이 있는데 여기선 사용 x
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # post-processing
        # decode_latents 사용
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

    # 결국 핵심을 다음 두 메소드임, 비디오 데이터에 맞게 처리
    def prepare_latent(self,
                       batch_size,
                       num_channels_latents,
                       video_length,
                       height,
                       width,
                       dtype,
                       device,
                       generator,
                       latents=None):
        # 만들어질 latent shape
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = 'cpu' if device.type == 'mps' else device

            if isinstance(generator, list):
                shape = shape
                latents = [torch.randn(shape, generator=generator[i], device=rand_device) for i in range(batch_size)]
                latents = torch.cat(latents, dim=0).to(device)

            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
            
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    
    def decode_latents(self, latents):
        # [B, C, F, H, W]
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        # 배치화 한뒤에 하나씩 decoding
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float().numpy()
        return video
    
        
    def _encode_prompt(self, 
                       prompt,
                       device,
                       num_videos_per_prompt,
                       do_classifier_free_guidance,
                       negative_prompt):
        """
        SDPL과 다르게 prmompt_embeds 등을 고려하지 않음
        무조건 prompt 하나인 상황을 가정
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        text_inputs = self.tokenizer(prompt,
                                     padding='max_length',
                                     max_length=self.tokenizer.model_max_length,
                                     truncation=True,
                                     return_tensors='pt')

        # 아래는 동일
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding='longest', return_tensors='pt').input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(text_input_ids.to(device),
                                            attention_mask=attention_mask)
        text_embeddings = text_embeddings[0]

        # SDPL 다음 코드 존재
        # prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # 차이점 : if do_classifier_free_guidance and negative_prompt_embeds is None:
        # uc embeeding을 얻도록 함
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # SDPL에선 if do_classifier_free_guidance 체크하고 다음을 수행
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)


            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    
    def check_inputs(self, prompt, height, width, callback_steps):
        """
        기존 보다 좀 더 간단, prompt 
        """
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_extra_step_kwargs(self, generator, eta):
        """
        SDPL과 동일
        """
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs