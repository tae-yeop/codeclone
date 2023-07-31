from typing import Any, Callable, Dict, List, Optional, Tuple, Union



from diffusers import StableDiffusionControlNetPipeline
from diffusers.models import ControlNetModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import is_compiled_module, logging, randn_tensor


def torch_dfs(model):
    result = [model]
    # modle의 children method를 활용
    for child in model.children():
        # recursion
        result += torch_dfs(child)
    # list를 리턴해서 append되도록 함
    return result


class StableDiffusionControlNetReferencePipeline(StableDiffusionControlNetPipeline):
    def prepare_ref_latents(self, refimage, ):

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        image=None,
        ref_image=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        guess_mode: bool = False,

    ):
        """
        guess_mode:
        prompt가 없더라도 인풋 이미지의 content를 알아내려고 한다?
        guidnace_Sacle과 연관된듯
        """

        assert reference_attn or reference_adain, "`reference_attn` or `reference_adain` must be True."

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
        )


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0


        # controlnet 준비
        # compiled 된 버전인지 확인 후 controlnet 알아옴
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet


        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (controlnet.config.global_pool_conditions
                                  if isinstance(controlnet, ControlNetModel)
                                  else controlnet.nets[0].config.global_pool_conditions)
        
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
        
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Preprocessing
        


        # 새로운 forward를 설정하도록 함
        if reference_adain:
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            # 깊어질수록 LayerNorm의 channel 값이 커진다
            # 낮은 채널 값으로 sorting하도록 함
            attn_modules = sorted(attn_modules, key=lambda x : -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))


        # 11. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_mdoel_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_mdoel_input, t)

                # 
                if guess_mode and do_classifier_free_guidance:
                    # random하게 뽑은 latent에 noise를 입힌다
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]

                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d)]) for d in donw_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # ref only
                noise = randn_tensor(ref_image_latents.shape, generator=generator,
                                     device=device, dtype=ref_image_latents.dtype)
                ref_xt = self.scheduler.add_noise(ref_image_latents,
                                                  noise,
                                                  t.reshape(1))
                ref_xt = self.scheduler.scale_model_input(ref_xt, t)

                # write할 때 ref_xt에 대해 기록
                MODE = 'write'
                self.unet(
                    ref_xt,
                    t,
                    text_encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )

                MODE = 'read'
                # controlnet 모델에서 얻은 skip conection을 연결해줌
                noise_pred = self.unet(
                    latent_mdoel_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)



    def pro
