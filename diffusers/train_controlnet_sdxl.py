import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from transformers import AutoTokenizer, PretrainedConfig

# SDXLPL에서 쓰는 방식으로
def encode_prompt(prompt_batch,
                  text_encoders,
                  tokenizers,
                  proportion_empty_propmts,
                  is_train=True):
    propmt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        # 랜덤하게 
        if random.random() < proportion_empty_propmts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # 여기서 random.choice?
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(captions,
                                    padding='max_length',
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device),
                                         output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            

    

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path:str,
                                               revision:str,
                                               subfolder:str="text_encoder"):
    """
    pretrained_model_name_or_path와 subfolder를 지정하면 해당 모델을 로딩
    """
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path,
                                                           subfolder=subfolder,
                                                           revision=revision)
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )

    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                                  subfolder="tokenizer", revision=args.revision, use_fast=False)
    # text enoder class 얻어옴
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision)
    # 개선된 vae 지정하려고 해둔듯
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )

    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    
    # controlnet 위치가 있다면
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    # 아니면 그냥 unet 그대로 가지고 옴, 어차피 UNet2DConditionModel 클래스이므로
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # SR 모듈 x
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Optimizer
    