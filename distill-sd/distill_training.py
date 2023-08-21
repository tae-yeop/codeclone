import argparse
import logging
import math
import gc
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


def prepare_unet(unet, model_type):
    """
    논문에 나온대로 Block-level elimination을 수행하는 부분
    """
    assert model_type in ["sd_tiny", "sd_small"]
    # Set mid block to None if mode is other than base
    # 코드에선 tiny이면 mid block을 없애고 small이라도 mnid가 있는거 같은데
    if model_type != "sd_small":
        unet.mid_block = None

    # Commence deletion of resnets/attentions inside the U-net
    # Handle Down Blocks
    for i in range(3):
        delattr(unet.down_blocks[i].resnets, "1")
        delattr(unet.down_blocks[i].attentions, "1")

    if model_type == "sd_tiny":
        delattr(unet.down_blocks, "3")
        unet.down_blocks[2].downsamplers = None

    else:
        delattr(unet.down_blocks[3].resnets, "1")
    # Handle Up blocks

    unet.up_blocks[0].resnets[1] = unet.up_blocks[0].resnets[2]
    delattr(unet.up_blocks[0].resnets, "2")
    for i in range(1, 4):
        unet.up_blocks[i].resnets[1] = unet.up_blocks[i].resnets[2]
        unet.up_blocks[i].attentions[1] = unet.up_blocks[i].attentions[2]
        delattr(unet.up_blocks[i].attentions, "2")
        delattr(unet.up_blocks[i].resnets, "2")
    if model_type == "sd_tiny":
        for i in range(3):
            unet.up_blocks[i] = unet.up_blocks[i + 1]
        delattr(unet.up_blocks, "3")
    torch.cuda.empty_cache()
    gc.collect()

unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )


# teach model 모델 만듬
KD_teacher_unet=UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

# sd_base는 없음
assert args.distill_level=="sd_small" or args.distill_level=="sd_tiny"

# 블록 elimination
prepare_unet(unet, args.distill_level)

# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)