from diffusers import StableDiffusionPipeline


def override_forward(self):
    def forward(sample,
                timestep,
                encoder_hidden_states,
                class_labels):



class DragPipeline(StableDiffusionPipeline):

    # forward 부분 수정
    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        prompt_embeds=None,
        batch_size=1,
        
    ):

        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps))):
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            noise_pred