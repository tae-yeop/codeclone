import argsparse
import inspect
from omegaconf import OmegaConf


import diffusers
import transformers

from accelerate.utils import set_seed
from accelerate.logging import get_logger

# Models
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
# LR 스케쥴러
from diffusers.optimization import get_scheduler

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline

from einops import rearrange


logger = get_logger(__name__, log_level="INFO")


def save_checkpoint(unet, mm_path):
    # unet에서 mm만 딱 빼서 저장하도록 함
    mm_state_dict = OrderedDict()
    state_dict = unet.state_dict()
    # 
    for key in state_dict:
        if 'motion_module' in key:
            mm_state_dict[key] = state_dict[key]
    torch.save(mm_state_dict, mm_path)


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
    # 기존 weights
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
    # 학습 안하는 부분 Freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # UNet에서 일단 전체 고정
    unet.requires_grad_(False)

    for name, module in unet.named_modules():
        # 이 부분에 해당하는 것만 학습하는 건데
        # down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q 이런식으로 나옴
        if 'motion_modules' in name and name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True
            
    # 항상 있는 코드 3개
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    
    # 8bit optim
    if use_8bit_adam:
        try:
            import bitsandbytes and bnb
        except ImportError:
        
        optimizer_cls = bnb.optim.AdamW8bit

    else:
        optimizer_cls = torch.optim.AdamW

    
    optimizer = optimizer_cls(
            unet.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )


    # Dataset 
    # huggingface Dataset이 아니라서?
    train_dataset = TuneAVideoDataset(**train_data)

    # 단 하나의 prompt밖에 없어서 이렇게 tokenization : Tuneavideo 코드에서 가져온게 분명
    train_dataset.propmt_ids = tokenizer(train_dataset.prompt,
                                         max_length=tokenizer.model_max_length,padding='max_length',
                                         truncation=True,
                                         return_tensors='pt').input_ids[0]
    
    # DataLoaders creation
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=train_batch_size)
    



    # valdiation pipeline
    # DDIM을 위한 noise scheduler 파라미터 넘기기
    validation_pipeline = AnimationPipeline(vae=vae, 
                                            text_encoder=text_encoder, tokenizer=tokenizer,
                                            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),)
    
    # VAE의 input slicing 활성화 : 처리할 때 메모리 소모를 줄이기 위해서 덜 배치화
    validation_pipeline.enable_vae_slicing()
    # ddim inv, Tuneavideo처럼
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler : 항상 있는 코드
    lr_scheduler = get_scheduler(lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
                                 num_training_steps=max_train_steps * gradient_accumulation_steps,)

    # Accelerator preparation
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 그냥 위에 prepare에 넣으면 안되나?
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    train_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Resume
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # 디렉토리 내부에서 checkpoint라는 이름으로 시작하는 녀석들 리스트화
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x : int(x.split('-')[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        # 여기서 checkpoint 로딩함
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-"))[1] # 파일 이름에 global_step 명시

        # epoch, step 재조정 (깔끔하게 맞아떨어지게)
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step * num_update_steps_per_epoch

    
    progress_bar = tqdm(range(global_step, max_train_step), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # resume step까진 그냥 pass
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # gradient accumulation 대비
            with accelerator.accumulate(unet):
                # 비디오 -> latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                # [B, F, C, H, W]
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, 'b f c h w -> (b f) c h w')
                # latent 얻을 떄 모두 이렇게 함
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                # scaling_factor
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # 각 비디오에서 랜덤한 timestep 샘플
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Forward process 수행
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # CLIP model을 통한 텍스트 임베딩 얻음
                encoder_hidden_states = text_encoder(batch['prompt_dis'])[0]

                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # Unet 전달
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # gather
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # checkpointing + val
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0


                if global_step % args.checkpointing_steps == 0 and accelerator.is_local_main_process:
            
                    save_path = os.path.join(
                        output_dir, f"checkpoint-{global_step}"
                    )
                    # 실제 save 부분
                    # 여긴 다양하게 사용
                    save_checkpoint(unet, save_path)
                    logger.info(f"Saved state to {save_path}")

                    # 이건 마지막 체크포인트만 남기는 옵션 : 용량이 부족하지 않게
                    if args.keep_only_last_checkpoint:
                        # Remove all other checkpoints
                        for file in os.listdir(args.output_dir):
                            if file.startswith(
                                "checkpoint"
                            ) and file != os.path.basename(save_path):
                                ckpt_num = int(file.split("-")[1])
                                if (
                                    args.keep_interval is None
                                    or ckpt_num % args.keep_interval != 0
                                ):
                                    logger.info(f"Removing {file}")
                                    shutil.rmtree(os.path.join(args.output_dir, file))

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)

                        ddim_inv_latent = None
                        if validation_data.use_inv_latent:
                            inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                            ddim_inv_latent = ddim_inversion()





