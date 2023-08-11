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



# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

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


# Create EMA for the unet.
if args.use_ema:
    ema_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

...

optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


dataset = ...


# 논문에서 언급한대로 augmentation
train_transforms = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

 def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


# DataLoaders creation:
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=args.train_batch_size,
    num_workers=args.dataloader_num_workers,
)


lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )


# Prepare everything with our `accelerator`.
unet, optimizer, train_dataloader, lr_scheduler,KD_teacher_unet = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler,KD_teacher_unet
    )
if args.use_ema:
    ema_unet.to(accelerator.device)


# Move text_encode and vae to gpu and cast to weight_dtype
text_encoder.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)


...

KD_teacher = {}
KD_student= {}

num_blocks= 4 if args.distill_level=="sd_small" else 3

# intermediate feature를 얻고자 한다 => 학습할 objective에 사용되기 때문
# forward hook에 걸리기 때문에 forwrad 이후 아웃풋을 activtion이라는 dictoionary에 저장
def getActivation(activation, name, residuals_present):
    if residuals_present:
        def hook(model, input, output):
            activation[name] = output[0]
    else:
        def hook(model, input, output):
            activation[name] = output

    return hook

# Hook을 거는 유틸
def cast_hook(unet, dicts, model_type, teacher=False):
    if teacher:
        for i in range(4):
            unet.down_blocks[i].register_forward_hook(getActivation(dicts, 'd'+str(i), True))
        unet.mid_block.register_forward_hook(getActivation(dicts, 'm', False))
        for i in range(4):
            unet.up_blocks[i].register_forward_hook(getActivation(dicts, 'u'+str(i), False))
    else: # student model
        num_blocks = 4 if model_type=='sd_small' else 3 #tiny이면 3개만
        for i in range(num_blocks):
            unet.down_blocks[i].register_forward_hook(getActivation(dicts, 'd'+str(i), True))

        # 이 부분 코드 이상한듯?
        if model_type=='sd_small':
            unet.mid_block.register_forwrad_hook(getActivation(dicts, 'm', False))
        for i in range(num_blocks):
            unet.up_blocks[i].register_forward_hook(getActivation(dicts, 'u'+str(i), False))

# hooking
cast_hook(unet,KD_student,args.distill_level,False)
cast_hook(KD_teacher_unet,KD_teacher,args.distill_level,True)


for epoch in range(first_epoch, args.num_train_epochs):
    unet.train()
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        ...
        with accelerator.accumulate(unet):
            # RGB -> VAE latent space
            latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor


            # latent에 입힐 Noise 설정
            noise = torchn.randn_like(latents)
            # 더 좋은 라이팅 효과를 위해서
            if args.noise_offset:
                noise += args.noise_offset * torch.randn((latetns.shape[0], latents.shape[1], 1, 1), device=latents.device)

            if args.


            # timestep
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz, ), device=latents.device)
            timesteps = timesteps.long()


            # 실제 noise를 적용하도록 한다
            if args.input_perturbation:
                noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Paramterziation에 따라서 target 값을 설정한다
            if args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


            # predict
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # teacher 쪽으로는 gradient 흐르지 않게 
            with torch.no_grad():
                teacher_pred=KD_teacher_unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Feature 별 loss 계산
            loss_features = 0
            if args.distill_level == 'sd_small':
                for i in range(4):

            else:


