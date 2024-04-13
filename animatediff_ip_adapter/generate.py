model_path = "runwayml/stable-diffusion-v1-5"
config_path = "config/prompts/01-ToonYou.json"
width = 512
height = 512
length = 16
device = 'cuda'
repeats = 1
xformers = False



# prompt_travel.json

#  "lora_map": {
#     "share/Lora/muffet_v2.safetensors" : 1.0,
#     "share/Lora/add_detail.safetensors" : 1.0
#   },

def load_safetensors_lora(text_encoder, unet, lora_path, alpha=0.75, is_animatediff=True):
    from safetensors.torch import load_file
    from animatediff.utils.lora_diffusers import (LoRANetwork,
                                                  create_network_from_weights)

    
    sd = load_file(lora_path)
    lora_network: LoRANetwork = create_network_from_weights(text_encoder, unet, sd, multiplier=alpha, is_animatediff=is_animatediff)
    lora_network.load_state_dict(sd, False)
    lora_network.merge_to(alpha)


    
def get_hf_pipeline(repo_id,
                    target_dir,
                    save=True,
                    force_download=False):
    ...
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=target_dir,
        local_files_only=True,
    )

    return pipeline
    
def get_base_model(model_name_or_path,
                   ):
    ...
    return Path(model_name_or_path)

def load_text_embedding(
    pipeline: DiffusionPipeline, text_embeds: Optional[tuple[str, torch.Tensor]] = None
):
    

def create_pipeline(
    base_model,
    model_config: ModelConfig = ...,
    infer_config: InferenceConfig = ...,
    use_xformers: bool = True,
) -> AnimationPipeline:

    motion_module = data_dir.joinpath(model_config.motion_module)

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder: CLIPSkipTextModel = CLIPSkipTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet: UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path=base_model,
        motion_module_path=motion_module,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")

    # set up scheduler
    sched_kwargs = infer_config.noise_scheduler_kwargs
    scheduler = get_scheduler(model_config.scheduler, sched_kwargs)


    # load the checkpoint weights into the pipeline
    if model_config.path is not None:
        model_path = data_dir.joinpath(model_config.path)
        if model_path.is_file():
            logger.debug("Loading from single checkpoint file")
            unet_state_dict, tenc_state_dict, vae_state_dict = get_checkpoint_weights(model_path)
        elif model_path.is_dir():
            logger.debug("Loading from Diffusers model directory")
            temp_pipeline = StableDiffusionPipeline.from_pretrained(model_path)
            unet_state_dict, tenc_state_dict, vae_state_dict = (
                temp_pipeline.unet.state_dict(),
                temp_pipeline.text_encoder.state_dict(),
                temp_pipeline.vae.state_dict(),
            )
            del temp_pipeline
        else:
            raise FileNotFoundError(f"model_path {model_path} is not a file or directory")

        # Load into the unet, TE, and VAE
        logger.info("Merging weights into UNet...")
        _, unet_unex = unet.load_state_dict(unet_state_dict, strict=False)
        if len(unet_unex) > 0:
            raise ValueError(f"UNet has unexpected keys: {unet_unex}")
        tenc_missing, _ = text_encoder.load_state_dict(tenc_state_dict, strict=False)
        if len(tenc_missing) > 0:
            raise ValueError(f"TextEncoder has missing keys: {tenc_missing}")
        vae_missing, _ = vae.load_state_dict(vae_state_dict, strict=False)
        if len(vae_missing) > 0:
            raise ValueError(f"VAE has missing keys: {vae_missing}")

        

    if use_xformers:
        logger.info("Enabling xformers memory-efficient attention")
        unet.enable_xformers_memory_efficient_attention()

    for l in model_config.lora_map:
        lora_path = data_dir.joinpath(l)
        if lora_path.is_file():
            load_safetensors_lora(text_encoder, unet, lora_path, alpha=model_config.lora_map[l])
        
    pipeline = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        controlnet_map=None,
    )

    # Load TI embeddings
    load_text_embeddings(pipeline)

    return pipeline
    

def run_inference(
    pipeline: AnimationPipeline,
    prompt: str = ...,
    
):
    


model_config = get_model_config(config_path)
# "motion_module": "models/motion-module/mm_sd_v15.ckpt"
is_v2 = is_v2_motion_module(model_config.motion_module)
infer_config: InferenceConfig = get_infer_config(is_v2)


controlnet_image_map, controlnet_type_map, controlnet_ref_map = controlnet_preprocess()
ip_adapter_map = ip_adapter_preprocess()


global g_pipeline

g_pipeline = create_pipeline(
    base_model = base_model_path,
    model_config = model_config,
    infer_config = infer_config,
    use_xformers = use_xformers
)


load_controlnet_models(pipe=g_pipeline, model_config=model_config)

# device 처리



# repeat the prompts if we're doing multiple runs
for _ in range(repeats):
    if model_config.prompt_map:
        prompt_map = {}
        for k in model_config.prompt_map.keys():
            ...


        output = run_inference(
            pipeline=g_pipeline,
            prompt="this is dummy string",
            n_prompt=n_prompt,
            
        )
        