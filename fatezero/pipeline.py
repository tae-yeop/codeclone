from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

def tensor_to_numpy(image, b=1):
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16

    image = image.cpu().float().numpy()
    image = rearrange(image, "(b f) c h w -> b f h w c", b=b)
    return image


class P2pSampleLogger:
    def __init__(self,
                 editing_prompts: List[str],
                 clip_length: int,):
        self.editing_prompts = editing_prompts
        self.clip_length = clip_length
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        

    def log_sample_images(self,
                          pipeline,
                          device,
                          step,
                          image=None,
                          latents=None,
                          uncond_embeddings_list=None,
                          save_dir=None):
        torch.cuda.empty_cache()
        samples_all = []
        attention_all = []

        if image is not None:
            input_pil_images = pipeline.numpy_to_pil(tensor_to_numpy(image))[0]
            if self.annotate:
                samples_all.append([annotate_image(image, "input seqeunce", font_size=self.annotate_size) for image in input_pil_images])

            else:
                samples_all.append(input_pil_images)

        for idx, prompt in enumerate(tqdm(self.))

        
class SpatioTemporalStableDiffusionPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(self,
                 vae,
                 text_encoder,
                 tokenizer,
                 unet,
                 scheduler: Union[DDIMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                LMSDiscreteScheduler,
                PNDMScheduler,]):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_before_train_loop(self, params_to_optimize=None):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        if params_to_optimize is not None:
            params_to_optimize.requires_grad = True

    def enable_vae_slicing(self):
        self.vae.enable_slicing()
    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def decode_latents(self, latents):
        is_video = (latents.dim() == 5)
        b = latents.shape[0]
        latents = 1 / 0.18215 * latents

        if is_video:
            latents = rearrange(latents, "b c f h w -> (b f) c h w") # torch.Size([70, 4, 64, 64])
        # batch 16개짜리로 나눠서 튜플얻음
        latents_split = torch.split(latents, 16, dim=0)
        # tuple로 나눠서 순차적으로
        image = torch.cat([self.vae.decode(l).sample for l in latents_split], dim=0)

        image = (image / 2 + 0.5).clamp(0, 1)
        
        image = image.cpu().float().numpy()
        if is_video:
            image = rearrange(image, "(b f) c h w -> b f h w c", b=b)
        else:
            image = rearrange(image, "b c h w -> b h w c", b=b)
        return image
    def prepare_latents(self, batch_size,
                    num_channels_latents,
                    clip_length,
                    height,
                    width,
                    dtype,
                    device,
                    generator,
                    latents=None,
                ):
        shape = (batch_size, num_channels_latents,
                 clip_length, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents is None:
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma


    @torch.no_grad()
    def __call__(self, **kwargs):
        edit_type = kwargs['edit_type']
        assert edit_type in ['save', 'swap', None]
        if edit_type is None:
        if edit_type == 'save':
        if edit_type == 'swap':
            return self.p2preplace_edit(**kwargs)