import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf
from dataclasses import dataclass

import torch
import torch.nn as nn

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps # 말그대로 임베딩, 타임 임베딩, 패치 임베딩, 이미지 임베딩 등

from diffusers.utils.import_utils import is_xformers_available

@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample : torch.FloatTensor

class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, 
                 flip_sin_to_cos: bool = True,
                 freq_shift: int = 0,
                 # 클래스 임베딩
                 class_embed_type: Optional[str] = None,
                 num_class_embeds: Optional[int] = None,

                 down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "DownBlock3D",),
                 mid_block_type: str = "UNetMidBlock3DCrossAttn",
                 up_block_type = ("UpBlock3D",
                    "CrossAttnUpBlock3D",
                    "CrossAttnUpBlock3D",
                    "CrossAttnUpBlock3D"),
                 only_cross_attention = False,
                 block_out_channels = (320, 640, 1280, 1280),
                 layers_per_block=2, # SDXL :2, SD1.5 :2
                 downsample_padding: int = 1, # SDXL, SD1.5 모두 1
                 
                 # 세부 정보
                 mid_block_scale_factor=1, # SDXL, SD1.5 모두 1
                 act_fn: str = "silu",
                 norm_num_groups: int = 32, # SDXL, SD1.5 모두 32
                 norm_eps = 1e-5,
                 cross_attention_dim: int = 1280, # SDXL : 2048, SD1.5 : 768
                 attention_head_dim = 8, # SDXL : [5,10,20], SD1.5 : 8
                 dual_cross_attention = False, # SDXL, SD1.5 : false
                 use_linear_projection = False, # SDXL :true, 
                 upcast_attention: bool = False, # SDXL, SD1.5 : false
                 resnet_time_scale_shift: str = "default", # SDXL, SD1.5 : "default"

                 # AnimateDiff 추가
                 use_motion_module              = False,
                motion_module_resolutions      = ( 1,2,4,8 ),
                motion_module_mid_block        = False,
                motion_module_decoder_only     = False,
                motion_module_type             = None,
                motion_module_kwargs           = {},
                unet_use_cross_frame_attention = None,
                unet_use_temporal_attention    = None,
                 ):

        super().__init__()

        # time

        time_embed_dim = block_out_channels[0] * 4 # 첫번째 block의 4배 만큼 time dim

        
        # 첫번째 block 채널과 일치
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedidng // 실제론 없음
        #  num_class_embed = null in SDXL
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == 'timestep':
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block =None
        self.up_blocks = nn.ModuleList([])

        # down block을 위한 준비
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)


        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2 ** i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block, # Final 아닐 때 마다 downsample함
                
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn = act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                # 추가
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            
        
    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded temporal unet's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        # 여기서 바꿈
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        from diffusers.utils import WEIGHTS_NAME

        # CONFIG_NAME = "config.json"
        # WEIGHTS_NAME = "diffusion_pytorch_model.bin"
        # FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
        # ONNX_WEIGHTS_NAME = "model.onnx"
        # SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
        # ONNX_EXTERNAL_WEIGHTS_NAME = "weights.pb"
        # HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"
        # DIFFUSERS_CACHE = default_cache_path
        # DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
        # HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
        # DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]
        # TEXT_ENCODER_ATTN_MODULE = ".self_attn"
        # 추가적인 config가 들어옴
        model = cls.from_config(config, **unet_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        # temporal 이 있다면
        params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
        print(f"### Temporal Module Parameters: {sum(params) / 1e6} M")
        
        return model

    def forward(self, sample,
                timestep, encoder_hidden_states, class_labels=None, attention_mask=None, return_dict=True):


        # 언제까지 foward 할지 결정
        # overall upsampling facotr의 곱까지 최소 forward됨
        
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        # 만약 사이즈가 맞아 떨어지지 않으면 
        # 디코더가 지원하는 사이즈 증가와 입력으로 들어온 샘플 사이즈가 맞는지 체크
        if any(s % default_overall_up_factor !=0 for s in sample.shpae[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # attention mask : 쓰이지 않음
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 원래라면 비슷하게 encoder attention mask 처리도 있음 : 쓰이지 않음
        

        # time : 이 부분은 기존의 contorlnet 코드와 같다
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == 'mps'
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        # 그냥 torch.tensor(100) 같은 경우
        elif len(timesteps.shape) == 0:
            # batch 축 하나 만듬
            timesteps = timesteps[None].to(sample.device)

        # sample의 배치만큼 늘려서 ONNXCoreML과 호환
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.

        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)


        # SDXL에선 class_embed_type=null
        # SD 1.5에선 class_embed_type가 없음
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == 'timestep':
                class_labels = self.time_proj(class_labels)
                
            class_emb = self.class_embeding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            # ca라면 attention mask가 있는게 차이인데 실제로 안쓰임
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_sample = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask # 어차피 none
                )
            else:
                # 기존과 다른점이 ca가 없어도 encoder_hidden_states가 들어감
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, 
                                                       encoder_hidden_states=encoder_hidden_states)

            down_block_res_samples += res_samples

        sample = self.mid_block(sample,emb,encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnet) : ]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                # 인코더의 중간 feature의 shape를 가지고 온다
                upsample_size = down_block_res_samples[-1].shape[2:]
                # upsample_size 이게 어떤 interpolation하는 역할을 하는 듯

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(hidden_states=sample,
                                        temb=emb,
                                        res_hidden_states_tuple=res_samples,
                                        encoder_hidden_states=encoder_hidden_states,
                                        upsample_size=upsample_size,
                                        attention_mask=attention_mask,)
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_sample,
                                        upsample_size=upsample_size, encoder_hidden_states=encoder_hidden_states,)

        # post
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
        
from tqdm import tqdm
from einops import rearrange
import torchvision
import imageio

# 비디오 프레임의 배치를 저장
def save_videos_grid(videos, path, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    # 매 timestep마다 grid를 생성
    for x in videos:
        # nrow : grid내의 row 갯수
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        # channel last로 변경
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


from safetensors import safe_open


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
parser.add_argument("--config",                type=str, required=True)

parser.add_argument("--L", type=int, default=16 )
parser.add_argument("--W", type=int, default=512)
parser.add_argument("--H", type=int, default=512)
args = parser.parse_args()


*_, func_args = inspect.getargvalues(inspect.currentframe())
func_args = dict(func_args)


# inference.yaml
# unet_additional_kwargs:
#   unet_use_cross_frame_attention: false
#   unet_use_temporal_attention: false
#   use_motion_module: true
#   motion_module_resolutions:
#   - 1
#   - 2
#   - 4
#   - 8
#   motion_module_mid_block: false
#   motion_module_decoder_only: false
#   motion_module_type: Vanilla
#   motion_module_kwargs:
#     num_attention_heads: 8
#     num_transformer_block: 1
#     attention_block_types:
#     - Temporal_Self
#     - Temporal_Self
#     temporal_position_encoding: true
#     temporal_position_encoding_max_len: 24
#     temporal_attention_dim_div: 1

# noise_scheduler_kwargs:
#   beta_start: 0.00085
#   beta_end: 0.012
#   beta_schedule: "linear"


# base: "models/DreamBooth_LoRA/moonfilm_reality20.safetensors"
# path: "models/DreamBooth_LoRA/lyriel_v16.safetensors"
# motion_module:
# - "models/Motion_Module/mm_sd_v14.ckpt"
# - "models/Motion_Module/mm_sd_v15.ckpt"
#   steps:          25


inference_config = OmegaConf.load(args.inference_config)
config = OmegaConf.load(args.config)

samples = []



motion_modules = model_config.motion_module
motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)

# 순서 : 모델 init => EMA => xformers
sample_idx = 0
for motion_module in motion_modules:
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path)
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    # 일단 3d 모델에 대해 init
    # to_container는 recursive하게 dict로 만들어줌
    unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

    # config로 init한 뒤에 항상 xformer 코드가 존재
    if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
    else: assert False

    # 어차피 inference니깐 cuda로 직접 보냄
    pipeline = AnimationPipeline(vae=vae,
                                 text_encoder=text_encoder,
                                 tokenizer=tokenizer,
                                 unet=unet, 
                                 scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))).to("cuda")
    
    motion_module_state_dict = torch.load(motion_module, mpa_location='cpu')
    # 만약 global step이 있으면
    # 먼저 motion module weigh를 가져옴
    if "global_step" in motion_module_state_dict:
        func_args.update({'global_step': motion_module_state_dict['global_step']})
        missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0

    # pipline에서 unet 부분 load
    # T2I 모델 가져옴
    if model_config.path != "":
        # ckpt라면
        if model_config.path.endswith(".ckpt"):
            state_dict = torch.load(model_config.path)
            pipeline.unet.load_state_dict(state_dict)
        # safetneosr라면 처리할게 많다
        elif model_config.endswith(".safetensors"):
            state_dict = {}
            # safe open으로 연다
            with safe_open(model_config.path, framework='pt', device='cpu') as f:
                # key 값에 맞는 tensor를 dict에 넣는듯
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

                # lora 처리
                # key에 lora가 있으면
                # base가 있는 경우가 LORA가 있는 경우일듯
                is_lora = all("lora" in k for k in state_dict.keys())
                if not is_lora:
                    base_state_dict = state_dict
                else:
                    base_state_dict = {}
                    with safe_open(model_config.base, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            base_state_dict[key] = f.get_tensor(key)

                # vae
                converted_vae_checkpoint = ...

    pipeline.to("cuda")

    prompts = model_config.prompt
    n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt

    random_seeds = model_config.get("seed", [-1])
    random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
    random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

    config[config_key].random_seed = []

    for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):


        sample = pipeline(prompts,
                          negative_prompt=n_prompts,
                          num_inference_steps = model_config.steps,
                          guidance_scale = model_config.guidance_scale,
                          width=args.W,
                          height=args.H,
                          video_length=args.L).videos
        samples.append(sample)

        prompt = "-".join()
        save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
        print(f"save to {savedir}/sample/{prompt}.gif")

        sample_idx += 1

samples = torch.concat(samples)
save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)
OmegaConf.save(config, f"{savedir}/config.yaml")