import os
import math
import wandb


from pathlib import Path


import diffusers

from animatediff.data.dataset import WebVid10M
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print

if __name__ == "__main__":
    parser = argparse.ArgumnetParser()
    main(**config)

def main(image_finetune,):
    ...
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    if not image_finetune:
        # training이면
        unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        # finetune 이라면
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    
    if unet_checkpoint_path != "":
        ...
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # finetune이면 trainable_modules는 그냥 "."이다.
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    trainale_params = list(filter(lambda p:p.requires_grad, unet.parameters()))
    optimizer = torch.optim.Adamw(trainable_params,
    lr=learning_rate,
    )

    # prepare acceleartor

    # dataset
    train_Dataset = WebVid10M(**train_data, is_image=image_finetune)

    # validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(unet=unet,
        vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()
    
    ...
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                ...
            
            pixel_values = batch["pixel_values"]
            video_length = pixel_values.shape[1]

            # vae encoding할 때 video인거 고려
            with torch.no_grad():
                if not image_finetune: #video라면 frame을 batchify한다
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else: # 그냥 이미지로 오면
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                
                latents = latents * 0.18215

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length, padding='max_length', truncation=True, 
                return_tensors='pt').input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
            
            if noise_scheduler.config.prediction_type == 'epsilon':
                target = noise
            elif noise_scheduler.config.prediction_type == 'v_prediction':
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            with torch.cuda.amp.autocast(enabled=mixed_prediction_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            if mixed_precision_training:
                scaler.sacle(loss).backward()
                # clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                # clip
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()

"""
validation_data:
  prompts:
    - "Snow rocky mountains peaks canyon. Snow blanketed rocky mountains surround and shadow deep canyons."
    - "A drone view of celebration with Christma tree and fireworks, starry sky - background."
    - "Robot dancing in times square."
    - "Pacific coast, carmel by the sea ocean and waves."
  num_inference_steps: 25
  guidance_scale: 8.

"""
            # validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                height = train_data.sample_size[0] if not ininstance(train_data.sample_size, int) else train_data.sample_size
                width = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                for idx, prompt in enumerate(prompts):
                    if not image_finetune:
                        sample = validation_pipeline(prompt, generator=generator,
                        video_length=train_data.sample_n_frames,
                        height=height,
                        width=width,
                        **validation_data).videos
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)
                    else:
                        sample = validation_pipeline(
                            
                        )

