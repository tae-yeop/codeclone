import argsparse
import inspect
from omegaconf import OmegaConf


import diffusers
import transformers

from accelerate.utils import set_seed
from accelerate.logging import get_logger

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline

from einops import rearrange


logger = get_logger(__name__, log_level="INFO")




def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "to_q",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,

    motion_module: str = "models/Motion_Module/mm_sd_v15.ckpt",
    inference_config_path: str = "configs/inference/inference.yaml",
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    inference_config = OmegaConf.load(inference_config_path)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                              mixed_precision=mixed_precision)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    # 스케쥴러
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    # text 관련
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    # vae
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder='vae')
    # unet
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

    # mm을 얻어옴
    motion_module_state_dict = torch.load(motion_module, map_location="cpu")
    if "global_step" in motion_module_state_dict: 
        func_args.update({"global_step": motion_module_state_dict["global_step"]})
    missing, unexpected = unet.load_state_dict(motion_module_state_dict, strict=False)
    assert len(unexpected) == 0
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    for name, module in unet.named_modules():
        if "motion_modules" in name and name.endswith(tuple(trainable_modules)):
            