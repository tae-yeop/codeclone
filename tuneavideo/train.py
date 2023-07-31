import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tuneavideo.models.unet import UNet3DConditionModel


#yaml에서 가져오는 module 값
# trainable_modules:
#   - "attn1.to_q"
#   - "attn2.to_q"
#   - "attn_temp"

def main(
    trainable_modules:Tuple[str] = ("attn1.to_q",
        "attn2.to_q",
        "attn_temp",)
    ):
    
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    