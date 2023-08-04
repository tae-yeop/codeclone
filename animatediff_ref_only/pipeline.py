# antimation pipe와 reference pipe를 참조하자
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from diffusers.utils import is_accelerate_available, BaseOutput, deprecate, randn_tensor, PIL_INTERPOLATION
from diffusers.configuration_utils import FrozenDict

from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
# ref image를 이용한 attention score 저장을 위해 
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models import AutoencoderKL
from ..models.unet import UNet3DConditionModel # AnimateDiff 모델
from transformers import CLIPTextModel, CLIPTokenizer

from einops import rearrange

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationReferencePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(self, 
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet3DConditionModel,
                 scheduler: Union[
                    DDIMScheduler,
                    PNDMScheduler,
                    LMSDiscreteScheduler,
                    EulerDiscreteScheduler,
                    EulerAncestralDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                ]):
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
    # refernce에서도 굳이 필요하지 않을듯함

    # DiffusinPipelne과 조금 다르다
    # 어차피 안쓸거임
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
    # device을 알아내는 유틸
    # cpu offload 상황을 고려해서 device 알아내는듯
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
                 # animatediff 관련
                 prompt:Union[str, List[str]],
                 video_length:Optional[int],
                 height:Optional[int]=None,
                 width:Optional[int]=None,
                 num_inference_steps:int=50,
                 guidance_scale:float=7.5,
                 negative_prompt:Optional[Union[str, List[str]]]=None,
                 eta:float=0.0,
                 generator:Optional[Union[torch.Generator, List[torch.Generator]]]=None,
                 latents:Optional[torch.FloatTensor]=None,
                 output_type:Optional[str]='tensor',
                 return_dict:bool=True,
                 callback:Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps:Optional[int]=1,

                 # animatediff 관련
                 num_videos_per_prompt:Optional[int]=1, # prompot 당 생성할 비디오 갯수
                 # ref only 관련
                 ref_image=None, # PIL Image
                 cross_attention_kwargs:Optional[Dict[str, Any]]=None,
                 attention_auto_machine_weight:float = 1.0, # hacked_forward에서 사용됨
                 style_fidelity:float = 0.5,
                 reference_attn: bool = True,
                 reference_adain: bool = True,

                 # 사용하지 않을 것들
                 # prompt_embeds: Optional[torch.FloatTensor] = None, 
                 # negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                 **kwrags):

        assert reference_attn or reference_adain, "`reference_attn` or `reference_adain` must be True."

    
        # 0. Default height and width to unet
        # height = height or self.unet.config.sample_size * self.vae_scale_factor
        # width = width or self.unet.config.sample_size * self.vae_scale_factor
        # reference 이미지에 맞춰서 height, width 설정
        height, width = self._default_height_width(height, width, ref_image)

        # 1. input 이상없는지 확인
        self.check_inputs(prompt, height, width, callback_steps)


        # 2. Define call parameters
        # referecen_only에선 prompt 여러개 사용하는 상황을 가정함
        # batch_size는 prompt갯수라고 봐야 할 듯
        batch_size = 1 # 여기서 batch_size가 생성할 이미지 갯수, prompt갯수 만큼
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # prompt trick
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        # negative 여러번
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size

        text_embeddings = self._encode_prompt(prompt, 
                                              device, 
                                              num_videos_per_prompt, 
                                              do_classifier_free_guidance, 
                                              negative_prompt)


        # 4. Preprocess reference image
        ref_image = self.prepare_image(
            image=ref_image,
            width=width,
            height=height,
            batch_size=batch_size * num_videos_per_prompt,
            num_videos_per_promptt=num_videos_per_prompt,
            device=device,
            dtype=text_embeddings.dtype,
        )

        
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
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


        # animatediff에서 따로 dtpye가지고 noise_pred의 dtype을 바꿔줌
        # latents_dtype = latents.dtype


        # 7. Prepare reference latent variables
        ref_image_latents = self.prepare_ref_latents(
            ref_image,
            batch_size * num_videos_per_prompt,
            text_embeddings.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
        
        # 8. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Modify self attention and group norm
        MODE = 'write'
        uc_mask = (
            torch.Tensor([1] * batch_size * num_images_per_prompt + [0] * batch_size * num_images_per_prompt)
            .type_as(ref_image_latents)
            .bool()
        )

        def hacked_basic_transformer_inner_forward(self, 
                                                   hidden_states: torch.FloatTensor,
                                                   attention_mask:Optional[torch.FloatTensor]=None,
                                                   encoder_hidden_states:Optional[torch.FloatTensor]=None,
                                                   encoder_attention_mask: Optional[torch.FloatTensor] = None,
                                                   timestep: Optional[torch.LongTensor] = None,
                                                   cross_attention_kwargs: Dict[str, Any] = None,
                                                   class_labels: Optional[torch.LongTensor] = None,
                                                   ):
            """
            첫번째 SA 리코딩
            animatediff에서 사용되는 basic_transformer는 : Transformer3DModel안에 있는 BasicTransformerBlock
            """
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # SA
            if MODE == 'write':
                self.bank.append(norm_hidden_states.detach().clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,)

            if MODE =='read':
                if attention_auto_machine_weight > self.attn_weight:
                    attn_output_uc = self.attn1(norm_hidden_states,
                                                encoder_hidden_states=torch.cat([norm_hidden_states]+self.bank, dim=1),
                                                **cross_attention_kwargs,)
                    attn_output_c = attn_output_uc.clone() 
                    if do_classifier_free_guidance and style_fidelity > 0:
                        attn_output_uc[uc_mask] = self.attn1(
                            norm_hidden_states[uc_mask],
                            encoder_hidden_states=norm_hidden_states[uc_mask],
                            **cross_attention_kwargs,
                        )
                    attn_output = style_fidelity * attn_output_c + (1.0 - style_fidelity) * attn_output_uc
                    self.bank.clear()

                else:
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )

            hidden_states = attn_output + hidden_states
            # CA
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # FF
            norm_hidden_states = self.norm3(hidden_states)
            ff_output = self.ff(norm_hidden_states)
            hidden_states = ff_output + hidden_states

            return hidden_states
        
        def hacked_mid_forward(self, *args, **kwargs):
            ...
        
        def hack_CrossAttnDownBlock2D_forward():
            ...

        def hacked_DownBlock2D_forward(self, hidden_states, temb=None):
            ...
        def hacked_CrossAttnUpBlock2D_forward():
            ...

        def hacked_UpBlock2D_forward():
            ...

        if reference_attn:
            # UNet에 있는 모든 BasicTransformerBlock에 대해서...
            # 사실상 BaisicTransformerBlock은 동일함
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

                
        if reference_adain:
            pass
        
        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


                # reference
                noise = randn_tensor(
                    ref_image_latents.shape, generator=generator, device=device, dtype=ref_image_latents.dtype
                )
                

    def _default_height_width(self, height, width, image):
        # from reference_only
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width
        
    def check_inputs(self, prompt, height, width, callback_steps):
        # from animatediff
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

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_videos_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
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

        # ref 이미지 갯수
        image_batch_size = image.shape[0]

        # 얼만큼 ref image를 복제할지 결정
        if image_batch_size == 1: # ref 이미지 한장이면
            repeat_by = batch_size # promt 갯수 * promt 당 비디오 갯수
        else:
            repeat_by = num_videos_per_prompt # 똑같이 batch_Size여야하는거 아닌가?ㄴ

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image
        
    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        # vae의 아웃풋으로 나오는 shape 형태로 맞춘다
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        # 미리 만들어둔 latents가 없기 떄문에 항상 다음은 실행됨
        if latents is None:
            # from SDPL
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # shape 체크
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # [B, C, F, H, W]
        return latents

    def prepare_ref_latents(self, refimage, batch_size, dtype, device, 
                            generator, do_classifier_free_guidance):

        # 여기서 refimage는 repeat_by x CFG를 위해 2배 된 상태가 아직 아님
        refimage = refimage.to(device=device, dtype=dtype)

        # 여기서 batch_size는 batch_size(prompt 배치사이즈) * num_videos_per_prompt
        if isinstance(generator, list):
            # generator에 맞춰서 이미지를 하나씩 latent space로 보내기
            # [i:i+1] 하는 이유는 배치 축 유지하기 위해서
            ref_image_latents = [self.vae.encode(refimage[i:i+1]).latent_dist.sample(generator=generator[i]) for i in range(batch_size)]

            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(generator=generator)

        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # 여기서 왜 다시 2배를 해야할까?, 앞에서 이미 prepare_image에서 이미 2배로 늘렸는데
        # 공식 코드에서 prepare_image을 실행할떄 CFG False 디폴트로 하고 넘어감
        if ref_image_latents.shape[0] < batch_size: # if isinstance(generator, list) 이 경우는 해당안됨
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            # propot 갯수만큼 늘려줌
            ref_image_latents = ref_image_latents.repeat(batch_size // ref_image_latents.shape[0], 1, 1, 1)

        # CFG니깐 2배 키워서 사용
        ref_image_latents = torch.cat([ref_image_latents] * 2) if do_classifier_free_guidance else ref_image_latents

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents



    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.
        latents = 1 / self.vae.config.scaling_factor * latents
        # batchfiy
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx: frame_idx+1].to(self.vae.device, self.vae.dtype)).sample)

        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video