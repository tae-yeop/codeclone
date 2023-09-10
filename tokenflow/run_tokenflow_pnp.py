


class TokenFlow(nn.Module):
    def __init__(self, config):
        self.config = config

        model_key = "runwayml/stable-diffusion-v1-5"
        
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained()
        self.scheduler.set_timesteps(config['n_timesteps'], device=self.device)

        

    def edit_video(self):
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        self.save_vae_recon()
        # 총 저장갯수 * pnp_f_t
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config['n_timesteps'] * self.config['pnp_attn_t'])

        self.init_method(conv_injection_t=pnp_f_t, qk_injection=pnp_attn_t)
        
        noisy_latents = self.scheduler.add_noise(self.latents, self.eps, self.scheduler.timesteps[0])
        edited_frames = self.sample_loop(noisy_latents, torch.arange(self.config['n_frames']))

        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_10.mp4')
        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_20.mp4', fps=20)
        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_30.mp4', fps=30)





    def sample_loop(self, x, indices):
        os.makedirs(f'{self.config}')
        for i, t in enumerate(self.scheduler.timesteps):
            x = self.batched_denoise_step(x, t, indices)

        decoded_latents = self.decode_latents(x)
        for i in range(len(decoded_latents)):
            T.ToPILImage()(decoded_latents[i]).save(f'{self.config["output_path"]}/img_ode/%05.png' % i)

        return decoded_latents


    @torch.autocast(dtype=torch.float16, device_type='cuda')
    def batched_denoise_step(self, x, t, indicies):
        batch_size = self.config['batch_size']
        denoised_latents = []
        pivotal_idx = torch.randint(batch_size, (len(x)//batch_size,)) + torch.arange(0, len(x), batch_size)

        register_pivotal(self, True)
        self.denoise_step(x[pivotal_idx], t, indices[pivotal_idx])
        register_pivotal(self, False)


        return denoised_latents


    def save_vae_recon(self):
        os.makedirs(f'{self.config["output_path"]}/vae_recon', exist_ok=True)
        decoded = self.decode_latents(self.latents)

        for i in range(len(decoded)):
            T.ToPILImage()(decoded[i]).save(f'{self.config["output_path"]}/vae_recon/%05d.png' % i)
        # fps 다르게 해서 저장    
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_10.mp4', fps=10)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_20.mp4', fps=20)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_30.mp4', fps=30)

    def init_method(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >=0 else []
        register_extended_attention_pnp(self, self.qk_injection_timesteps)
        register_conv_injection(self, self.conv_injection_timesteps)
        set_tokenflow(self.unet)

def run(config):
    seed_everythin(config['seed'])
    editor = TokenFlow(config)
    editor.edit_video()


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='configs/config_pnp.yaml')
opt = parser.parse_args()
with open(opt.config_path, "r") as f:


run(config)