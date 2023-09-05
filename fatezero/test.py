


def collate_fn(examples):
    batch = {
        "prompt_ids": torch.cat([example["prompt_ids"] for example in examples], dim=0),
        "images": torch.stack([example["images"] for example in examples]),
    }
    return batch

def test():
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretraiend(pretrained_model_path,
                                              subfolder="tokenizer",
                                              use_fast=False)
    text_encoder = CLIPTextModel.from_pretraiend(pretrained_model_path,
                                                 subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path,
                                        subfolder="vae")
    
    unet = UNetPseudo3DConditionModel.from_pretrained(
        os.path.join(pretrained_model_path, "unet"), model_config=model_config
    )

    #'SpatioTemporalStableDiffusionPipeline'
    # 'P2pDDIMSpatioTemporalPipeline'
    pipeline = P2pDDIMSpatioTemporalPipeline(test_pipeline_config,
                                             vae=vae,
                                             text_encoder=text_encoder,
                                             tokenizer=tokenizer,
                                             unet=unet,
                                             scheduler=DDIMScheduler.from_pretrained(
                                                pretrained_model_path,
                                                subfolder="scheduler",
                                            ),)

    pipeline.scheduler.set_timesteps(editing_config['num_inference_steps'])
    pipeline.set_progress_bar_config(disable=True)
    pipeline.print_pipeline(logger)

    
    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    prompt_ids = tokenizer(dataset_config["prompt"],
                           truncation=True,
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           return_tensors="pt",).input_ids

    video_dataset = ImageSequenceDataset(**dataset_config, prompt_ids=prompt_ids)
    train_dataloader = torch.utils.data.DataLoader(video_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    unet, train_dataloader = accelerator.prepare(unet, train_dataloader)

    weight_dtype = torch.float32

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, weight_dtype)

    ...
    validation_sample_logger = P2pSampleLogger(**editing_config, logdir=logdir, source_prompt=dataset_config['prompt'])
    
    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    batch = next(train_data_yielder)

    
    vae.eval()
    text_encoder.eval()
    unet.eval()

    images = batch['images'].to(dtype=weight_dtype)
    images = rearrange(images, "b c f h w -> (b f) c h w")

    if accelerator.is_main_process:
        



def run():
    Omegadict = OmegaConf.load(config)
    if 'unet' in os.listdir(Omegadict[''])