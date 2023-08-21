# Load scheduler, tokenizer and models.
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)

...

if args.use_peft:
    from peft import LoraConfig, LoraModel, get_peft_model_state_dict, set_peft_model_state_dict

    UNET_TARGET_MODULES =["to_q", "to_v", "query", "value"]
    TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]

    config = LoraConfig(r=args.lora_r, lora_alphas=args.lora_alpha, 
                        target_modules=UNET_TARGET_MODULES,
                        lora_dropout=args.lora_dropout,
                        bias=args.lora_bias)
    unet = LoraModel(config, unet)

    vae.requires_grad_(False)
    if args.train_text_encoder:
        config = LoraConfig(r=args.lora_text_encoder_r,
                            lora_alpha=args.lora_text_encoder_alpha,
                            target_modules=TEXT_ENCODER_TARGET_MODULES,
                            lora_dropout=args.lora_text_ecoder_dropout,
                            bias=args.lora_text_encoder_bias)
        text_encoder = LoraModel(config, text_encoder)

else:
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

vae.to(accelerator.device, dtype=weight_dtype)
if not args.train_text_encoder:
    text_encoder.to(accelerator.device, dtype=weight_dtype)


if args.use_peft:
    params_to_optimize = (itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters())

    optimizer = optimizer_cls(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
else:
    optimizer = optimizer_cls(
            lora_layers.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )


dataset = ...

...
# DataLoaders creation:
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=args.train_batch_size,
    num_workers=args.dataloader_num_workers,
)

overrod_max_train_steps = False
lr_scheduler = get_scheduler(args.lr_scheduler,
                             optimizer=optimizer,
                            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                            num_training_steps=args.max_train_steps * accelerator.num_processes,
                        )

if args.use_peft:
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
else:
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            lora_layers, optimizer, train_dataloader, lr_scheduler
        )


if accelerator.is_main_process:
    accelerator.init_trackers("text2image-fine-tune", config=vars(args))


for epoch in range(first_epoch, args.num_train_epochs):
    unet.train()
    if args.train_text_encoder:
        text_encoder.train()
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        ...
        with accelerator.accumulate(unet):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if args.use_peft:
                    params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                else:
                    params_to_clip = lora_layers.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

# Save the lora layers
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    # peft를 쓴다면 다음과 같이
    if args.use_peft:
        lora_config = {}
        unwarpped_unet = accelerator.unwrap_model(unet)
        state_dict = get_peft_model_state_dict(unwarpped_unet, state_dict=accelerator.get_state_dict(unet))
        lora_config["peft_config"] = unwarpped_unet.get_peft_config_as_dict(inference=True)
        
        if args.train_text_encoder:
            unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder_state_dict = get_peft_model_state_dict(
                    unwarpped_text_encoder, state_dict=accelerator.get_state_dict(text_encoder)
                )
            text_encoder_state_dict = {f"text_encoder_{k}": v for k, v in text_encoder_state_dict.items()}


            
        accelerator.save(state_dict, os.path.join(args.output_dir, f"{global_step}_lora.pt"))
        with open(os.path.join(args.output_dir, ))

        
    else:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)   


# Final inference
# Load previous pipeline
pipeline = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
)

if args.use_peft:
    def load_and_set_lora_ckpt(pipe, ckpt_dir, global_step, device, dtype):
        with open(os.path.join(args.output_dir, f'{global_step}_lora_config.json'), 'r') as f:
            lora_config = json.load(f)
        print(lora_config)

        checkpoint = os.path.join(args.output_dir, f"{global_step}_lora.pt")
        lora_checkpoint_sd = torch.load(checkpoint)

        unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k}
        text_encoder_lora_ds = {
                k.replace("text_encoder_", ""): v for k, v in lora_checkpoint_sd.items() if "text_encoder_" in k
            }

        unet_config = LoraConfig(**lora_config["peft_config"])
        pipe.unet = LoraModel(unet_config, pipe.unet)
        set_peft_model_state_dict(pipe.unet, unet_lora_ds)

        
        if "text_encoder_peft_config" in lora_config:
            text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
            pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder)
            set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)
            
        if dtype in (torch.float16, torch.bfloat16):
            pipe.unet.half()
            pipe.text_encoder.half()
                
        pipe.to(device)
        return pipe


    pipeline = load_and_set_lora_ckpt(pipeline, args.output_dir, global_step, accelerator.device, weight_dtype)
else:
    pipeline = pipeline.to(accelerator.device)
    # load attention processors
    pipeline.unet.load_attn_procs(args.output_dir)

if args.seed is not None:
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
else:
    generator = None
images = []
for _ in range(args.num_validation_images):
    images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

if accelerator.is_main_process:
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add
        if tracker.name == 'wandb':
            tracker.log({'test':[wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                 for i, image in enumerate(images)]})


accelerator.end_training()
