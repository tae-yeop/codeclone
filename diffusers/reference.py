# https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_reference.py

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

# SDPL을 상속해서 사용
from diffusers import StableDiffusionPipeline
# ref image를 attention layer에 나온 결과를 활용해야함
from diffusers.models.attention import BasicTransformerBlock
# 각 블록의 forward를 조작하도록 함
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
# 기존 SDPL과 같은 아웃풋 형식
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
# guidance_rescale에 맞춰서 noise 리스케일
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.utils import PIL_INTERPOLATION, logging, randn_tensor 
# PIL_INTERPOLATION = {
#         "linear": PIL.Image.LINEAR,
#         "bilinear": PIL.Image.BILINEAR,
#         "bicubic": PIL.Image.BICUBIC,
#         "lanczos": PIL.Image.LANCZOS,
#         "nearest": PIL.Image.NEAREST,
#     }
# diffuser에도 logging이 있다
# rand_tensor : Device, Generraotr, Data Type, Layout, Error Handling and Logging Control 가능
# 호환 가능한 N(0,1) tensor만듬

logger = logging.get_logger(__name__)

# depth first search : pass한 모델 아래의 모든 Module을 다 가지고 옴
def torch_dfs(model):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class StableDiffusionReferencePipeline(StableDiffusionPipeline):
    def _default_height_width(self, height, width, image):
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            height = (height // 8)*8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

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
        guess_mode=False, # 이게 뭘까?
    ):  
        """
        ref image를 preprocessing(concat)
        만약 ref_image [Num, C, H, W]이면 [Num * repeat, C, H, W]로 만듬
        repeat는 prompt 하나에 이미지 얼마나 만들어낼지에 따라 
        """
        # 이미지 형식에 맞춰 처리
        # 아마 직접적인 텐서가 아니라 리스트이면
        if not isinstance(image, torch.Tensor):
            # PIL 이면 리스트화시킴
            if isinstance(image, PIL.Image.Image):
                image = [image]

            
            if isinstance(image[0], PIL.Image.Image):
                # PIL -> numpy -> tensor
                images = []
                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)
                    
                image = images
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)

            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        # 만약 그냥 텐서이면 바로 batch size 얻음
        image_batch_size = image.shape[0]
        # reference 이미지를 여러개 배치화 시킴
        # 이미지 한장이면 한번만
        if image_batch_size == 1:
            repeat_by = batch_size # promt 갯수만큼 반복
        # ref 이미지가 여러개라면 prompt당 여러번 생성함
        else:
            repeat_by = num_images_per_prompt
        # 갯수 반복함
        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        # guess mode를 안쓴다면, 다시 2배로 늘림
        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_ref_latents(self, refimage, batch_size, dtype, device, generator, do_classifier_free_guidance):
        """
        여기서 batch_size는 총 생성할 이미지 갯수 = batch_size * num_images_per_prompt
        """
        refimage = refimage.to(device=device, dtype=dtype)

        # 마스크 이미지를 latent space로 임베딩
        if isinstance(generator, list): # seed에 따라 여러개 만든다면
            ref_image_latents = [self.vae.encode(refimage[i: i+1]).latent_dist.sample(generator=generator[i]) for i in range(batch_size)]

            ref_image_latents = torch.cat(ref_image_latents, dim=0)
            
        else: # seed 하나뿐이면
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(generator=generator)
            
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # 마스크를 복제, 각 promp
        if ref_image_latents.shape[0] < batch_size:
            # ref 이미지를 3개 넣으면 batch size는 3의 배수가 되어야 함.
            # 이게 맞아떨어져야 다음번 반복하는 코드가 실행됨
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError()

            # 총 생성이미지 갯수/ref image batch 축 
            ref_image_latents = ref_image_latents.repeat(batch_size // ref_image_latents.shape[0], 1, 1, 1)

        # 만약 CFG를 쓰면 복제 아니면 그대로
        ref_image_latents = torch.cat([ref_image_latents] * 2) if do_classifier_free_guidance else ref_image_latents

        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        ref_image=None, # PIL Image
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator:Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents=None,
        prompt_embeds:Optional[torch.FloatTensor]=None,
        negative_prompt_embeds:Optional[torch.FloatTensor]=None,
        return_dict:bool = True,
        callback:Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps:int = 1,
        cross_attention_kwargs:Optional[Dict[str, Any]]=None,
        attention_auto_machine_weight:float = 1.0, # hacked_forward에서 사용됨
        style_fidelity:float = 0.5,
        reference_attn: bool = True,
        reference_adain: bool = True,
        
    ): 
        """
        num_images_per_prompt:
        prompt 한개당 생성할 이미지 갯수
            
        attention_auto_machine_weight : 
        self attention에서 reference query에 대한 가중치
        1.0이면 refernce query를 모든 self attention의 context에 사용됨

        style_fidelity:
        prompt을 우선할지(0.0) control을 우선할지 결정(1.0)
        ref_uncond_xt의 style_fidelity.


        callback:
        inference중에 불려지는 callback
        callback(step: int, timestep: int, latents: torch.FloatTensor) 형식의 함수 
        
        callback_steps:
        callback을 몇번 call할지
        """
        # 둘 중에 하나는 무조건 true
        assert reference_attn or reference_adain, "`reference_attn` or `reference_adain` must be True."

        # 0. refrence 이미지의 h,w 리턴
        height, width = self._default_height_width(height, width, ref_image)

        # 1. input 이상없는지 확인
        # SDPL과 동일
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)
        

        # 2. Define call parameters
        # prompt에 따른 길이
        # 만약 prompt 한개이면 batch_size 1
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        # 여러개 prompt가 들어오면 여기에 batch_size를 맞춘다
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # 7.5인경우 CFG 사용
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt : SDPL와 동일
        # cross_attention_kwargs있다면 lora scale에 None을 넣도록 함
        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        # LoRA 레이어가 로드되었ㄷ면 text 인코더에 모든 LoRA 레이어에 lora scale을 적용
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt,
                                            do_classifier_free_guidance,
                                            negative_prompt,
                                            prompt_embeds=prompt_embeds,
                                            negative_prompt_embeds=negative_prompt_embeds,
                                            lora_scale=text_encoder_lora_scale,)

        # 4. reference 이미지 preprocessing
        ref_image = self.prepare_image(
            image=ref_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt, # prompt 갯수 x prompt당 생성 이미지 갯수
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        # 여기서 ref_image는 [반복 생성할 갯수*prompt 갯수, C, H, W]

        # 5. Prepare timesteps : SDPL 동일
        # 50번 step 셋팅
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps # 10번 과정 : denosing에서 사용

        # 6. Prepare latent variables : SDPL 사용
        num_channels_latents = self.unet.config.in_channels
        # batch_size * num_images_per_prompt = 총 생성 이미지 갯수
        latents = self.prepare_latents(batch_size * num_images_per_prompt,
                                       num_channels_latents,
                                       height,
                                       width,
                                       prompt_embeds.dtype,
                                       device,
                                       generator,
                                       latents,)

        
        # 7. Prepare reference latent variables
        # VAE를 이용해서 ref image -> latent space
        ref_image_latents = self.prepare_ref_latents(
            ref_image,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
        # ref_imag_latents : [반복 생성할 갯수*prompt 갯수, C_latent, 64, 64]

        # 8. Prepare extra step kwargs : 기존 SDPL 동일, 뭔가 추가 작업할게 있다면 수행
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. SA와 GroupNorm 변경
        MODE = 'write'
        # [True,True,True, ... False, False, False]
        uc_mask = torch.Tensor([1] * batch_size * num_images_per_prompt + [0] * batch_size * num_images_per_prompt).type_as(ref_image_latents).bool()
    
        # 변경된 forward 정의
        # 먼저 transformer block 부터
        def hacked_basic_transformer_inner_forward(self, 
                                                   hidden_states: torch.FloatTensor,
                                                    attention_mask: Optional[torch.FloatTensor] = None,
                                                    encoder_hidden_states: Optional[torch.FloatTensor] = None,
                                                    encoder_attention_mask: Optional[torch.FloatTensor] = None,
                                                    timestep: Optional[torch.LongTensor] = None,
                                                    cross_attention_kwargs: Dict[str, Any] = None,
                                                    class_labels: Optional[torch.LongTensor] = None,
                                                ):

            # 이부분은 기존의 BaiscTransformer forward와 일치
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}


            # cross만 사용하는지 체크
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else: # 보통 SA가 있으니깐 이걸 실행될텐데
                if MODE == 'write':
                    self.bank.append(norm_hidden_states.detach().clone()) # norm 결과를 저장해둠
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == 'read': # read 상황에서
                    # query weight 보다 작다면
                    if attention_auto_machine_weight > self.attn_weight:
                        # 더이상 SA가 아님
                        attn_output_uc = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                            **cross_attention_kwargs,
                        ) # write에서 얻은 bank을 encoder로 넣음
                        attn_output_c = attn_output_uc.clone() 
                        if do_classifier_free_guidance and style_fidelity >0: # style 우선
                            attn_output_c[uc_mask] = self.attn1()

                        attn_output = style_fidelity * attn_output_c + (1.0 - style_fidelity) * attn_output_uc
                        self.bank.clear()
                    else:
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs
                        )

            # 원래 BasicTrasnformer에선 첫번쨰 SA 이후에 이게 실행됨
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states
            # 뒷 부분은 똑같음
            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        def hacked_mid_forward(self, *args, **kwargs):
            eps=1e-6
            x = self.original_forward(*args, **kwargs)

            if MODE == 'write':
                if gn_auto_machine_weight >= self.gn_weight:
                    
        def hacked_UpBlock2D_forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            eps = 1e-6
            for i, resnet in enumerate(self.resnets):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                # channle 방향
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if MODE == 'write':
                    if gn_auto_machine_weight >= self.gn_weight:
                        # adain에서 사용할 평균 분산을 bank에 기록
                        var, mean = torch.var_mean(hidden_states, dim=(2,3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == 'read':
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2,3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps)**0.5 # clipping
                        mean_acc = sum(self.mean_bank[i]  / float(len(self.mean_bank[i])))
                        var_acc = sum(self.var_bank)
                        
        # attn = True이면
        if reference_attn:
            # 일단 TransformerBlock 전부 다 찾는다
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            # attenton module에서 nrom1의 batch 기준으로 sorting
            # 그런데 BasicTransformerBlock에는 normalized_shape이 없는거 처럼 보이는데?
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            # 원본 forward를 hacked 버전으로 대체함
            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                # attnetion weight가 아마 후반에 있는 모듈일 수도록 점점 1.0에 가까워짐
                module.attn_weight = float(i) / float(len(attn_modules))

        # adain=True이면
        if reference_adain:
            # gn block을 먼저 모으도록 함
            gn_modules = [self.unet.mid_block]
            self.unet.mid_block.gn_weight = 0

            # down  + up 블록들도 수집
            down_blocks = self.unet.down_blocks
            # 가중치는 깊어질수록 낮아지게
            for w, module in enumerate(down_blocks):
                module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                gn_modules.append(module)
            up_blocks = self.unet.up_blocks
            # 가중치 점점 높아지게
            for w, module in enumerate(up_blocks):
                module.gn_weight = float(w) / float(len(up_blocks))
                gn_modules.append(module)

            for i, module in enumerate(gn_modules):
                if getattr(module, "original_forward", None) is None:
                    module.original_forward = module.forward
                if i == 0:
                    # mid block
                    module.forward = hacked_mid_forward.__get__(module, torch.nn.Module)
                elif isinstance(module, CrossAttnDownBlock2D):
                    module.forward = hack_CrossAttnDownBlock2D_forward.__get__(module, CrossAttnDownBlock2D)
                elif isinstance(module, DownBlock2D):
                    module.forward = hacked_DownBlock2D_forward.__get__(module, DownBlock2D)
                elif isinstance(module, CrossAttnUpBlock2D):
                    module.forward = hacked_CrossAttnUpBlock2D_forward.__get__(module, CrossAttnUpBlock2D)
                elif isinstance(module, UpBlock2D):
                    module.forward = hacked_UpBlock2D_forward.__get__(module, UpBlock2D)

                module.mean_bank = []
                module.var_bank = []
                module.gn_weight *= 2

        # 9. Denosing loop 기본 SDPL과 똑같은듯?
        # self.scheduler.order는 무엇일까?
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_Bar:
            for i, t in enumerate(timesteps):
                # CFG를 사용한담녀 latents를 늘린다
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # noise를 입힌다
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # ref latent에 대해
                noise = randn_tensor(ref_image_latents.shape, generator=generator, device=device, dtype=ref_image_latents.dtype
                )
                
                ref_xt = self.scheduler.add_noise(ref_image_latents, noise, t.reshape(1,))
                ref_xt = self.scheduler.scale_model_input(ref_xt, t)

                MODE = 'write'
                # 
                self.unet(ref_xt, t, encoder_hidden_states=prompt_embeds,
                          cross_attention_kwargs=cross_attention_kwargs,
                          return_dict=False,)

                # noise residual 예측
                MODE = 'read'
                noise_pred = self.unet(latent_model_input,
                                       t,
                                       encoder_hidden_states=prompt_embeds,
                                       cross_attention_kwargs=cross_attention_kwargs,
                                       return_dict=False)[0]

                # 아래 부터는 SDPL과 동일
                # CFG : SDPL과 동일
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_sclae * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                    
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # SDPL 과 동일
        # image를 리턴한다면
        if not output_type == 'latent':
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # self.image_processor = VaeImageProcessor 
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)