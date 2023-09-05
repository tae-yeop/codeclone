    
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    logging,
    randn_tensor,
    replace_example_docstring,)

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from diffusers.utils import BaseOutput, randn_tensor
from .arcface.model import IDLoss
from .face_id_utils import get_tensor_M

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class StableDiffusionControlNetFreedomPipelineOutput(BaseOutput):
    images : Union[List[PIL.Image.Image], np.ndarray]
    intermediates : Union[Dict, np.ndarray]
        


class StableDiffusionControlNetFreedomPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        freedom_condition_type='face_id',
        guide_weight_path='',
    ):
        super().__init__()

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        if freedom_condition_type == 'face_id':
            freedom_energy_guide = IDLoss(weight_path=guide_weight_path)
        elif freedom_condition_type == 'style':
            # freedom_energy_guide = CLIPImageModel(need_ref=True, ref_path=add_ref_path)
            ...
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            freedom_energy_guide=freedom_energy_guide
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)


    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype
            
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
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

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
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

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        # vae로 넣기전에 전처리
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        # [batch_size * cfg_2, 3, height, weight], 범위는 0~1사이값
        return image
        
    def prepare_latent(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
            
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt, # prompt 
        image: PipelineImageInput = None, # condition image -> controlnet
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        repeat: int = 1,
        freedom_start: int = 40,
        freedom_end: int = -10,
        reference_img: PipelineImageInput = None,
        strength: float = 1.0
        ):

        r"""

        controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
            controlnet 아웃풋이 unet residual로 더해지기 전에 scale걸기

        control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
            controlnet을 전체 step에서 어디부터 적용할지 

        
        repeat ('int'):
            time travel strategy

        freedom_start ('int', defaults to 40):
            freedom을 적용하기 시작하는 timestep
            0 이하이면 freedom을 적용하지 않음
        freedom_end ('int', defaults to -10):
            freedom
            0 이하이면 denosiing이 끝날 때 까지 수행

        reference_img:
            freedom을 위한 reference images

        strength:
            control 강도 조절
        """
        # is_compiled_module?
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) :
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        ...
            
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0


        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # from freedom
        controlnet_conditioning_scale = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13) 

        # 3. Encode input prompt
        text_encoder_lora_scale = (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. prepare image
        # condition 이미지를 전처리하는 부분
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]

            reference_img = self.prepare_image(
                image=reference_img,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
                
            )

            
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt,
                                       num_channels_latents,
                                       height,
                                       width,
                                       prompt_embeds.dtype,
                                       device,
                                       generator,
                                       latents,
                                       )
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        # 전체 timestep 수 =  1000 부터 0 까지
        # timestep마다 keeps 리스트를 저장함
        # 여긴 scale과 별로도 keep할지 여부인것으로 보임
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i+1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)


        # 7.2 reference image 
        M = get_tensor_M(reference_path)
        self.grid = F.affine_grid(M, (1, 3, 256, 256), align_corners=True)

            
        
        # 8. Denosing loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Prepare computing energy guidance
                latents.requires_grad_ = True
                self.unet.requires_grad_(True)
                self.controlnet.requires_grad_(True)

                for j in range(repeat):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # controlnet(s) inference
                    # 만약 guess mode이면서 CFG라면
                    if guess_mode and do_classifier_free_guidance:
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1] # cfg condition만 가져옴
                    else:
                        control_model_input = latent_model_input # controlnet이 같이 들어가는 latent
                        controlnet_prompt_embeds = prompt_embeds

                    # scale 값 설정
                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )


                    noise_pred = self.unet(latent_model_input,
                                        t,
                                        encoder_hidden_states=prompt_embed)
                    

                    # CFG 처리
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        correction = noise_pred_text - noise_pred_uncond
                        noise_pred = noise_pred_uncond + guidance_scale * correction

                    alphas = self.scheduler.alphas_cumprod
                    alphas_prev = ...
                    sigmas = ...
                    sqrt_one_minus_alphas = ...


                    a_t = torch.full((batch_size * num_images_per_prompt, 1, 1, 1))
                    a_prev = torch.full((batch_size * num_images_per_prompt, 1, 1, 1), alphas_prev[t], device=device)

                    beta_t = a_t / a_prev
                    sigma_t = torch.full((batch_size * num_images_per_prompt, 1, 1, 1), sigmas[t], device=device)
                    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[t], device=device)

                    pred_x0 = (latents - sqrt_one_minus_at) / a_t.sqrt()

                    # 실제론 False가 되서 실행 안됨
                    if quantize_denoised:
                        pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

                    # apply freedom guidaance
                    if freedom_start > t >= freedom_end:
                        # 이미지 얻고
                        D_x0_t = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                        # warping
                        warp_D_x0_t = F.grid_sample(D_x0_t, self.grid, align_corners=True)
                        residual = self.idloss.get_residual(warp_D_x0_t)
                        norm = torch.linalg.norm(residual)
                        norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
                        rho = (correction * correction).mean().sqrt().item() * guidance_scale
                        rho = rho / (norm_grad * norm_grad).mean().sqrt().item() * 0.08  # 0.08

                    c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
                    c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
                    c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
                    c3 = (c3.log() * 0.5).exp()
                    x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

                    if end <= t < start:
                        x_prev = x_prev - rho * norm_grad.detach()

                    # repeat_noise = False가 들어감
                    x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * randn_tensor(x.shape)


                latents = x_prev.detach()
                prev_x0_image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                intermediates['x_inter'].append(prev_x0_image)
                intermediates['pred_x0'].append(pred_x0.detach())

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents


        return StableDiffusionControlNetFreedomPipelineOutput(imgaes=image, intermediates=intermediates)