


class DDIMSampler(object):
    def __init__(self, model, schedule='linear',
                 add_condition_mode='face_id',
                 ref_path=None, add_ref_path=None, no_freedom=False, **kwargs):
        
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps



    @torch.no_grad()
    def sample(self,
               S,
               x0=None,
               score_corrector=None):
        


        samples, intermediates = self.ddim_sampling(conditioning,
                                                    size,
                                                    x_T = x_T,
                                                    x0 = x0,
                                                    score_corrector=score_corrector)


    @torch.no_grad()
    def ddim_sampling(self, cond, shape, x_T=None, x0=None, score_corrector=None
                      ):
        device = self.model.betas.device

        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)


        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {'x_inter': [img], 'pred_x0': [img]}


        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim_pose(img, cond,
                                           ts, index=index,
                                           score_corrector=score_corrector)

            img, pred_x0 = outs
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)

    

    # step 한번 밟는 메소드
    def p_sample_ddim_pose(self, x, c, t, index,
                           score_corrector=None):
        b, *_, device = *x.shape, x.device


        # codntion을 requires_grad = True로
        x.requires_grad = True
        self.mdeol.requires_grad_(True)

        repeat = 1 # time-travel strategy
        start = 40
        end = -10

        for j in range(repeat):

            # CFG 단계
            if unconditional_conditioning is None or unconditional_guidance_sacle == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction


            # 모델에서 나온 x_t
            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == 'eps', 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            
            alphas = slef.model.alpha_cumprod if use_original_steps else self.ddim_alphas
            alpha_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev

            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas

            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)

            
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1,1,1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # x_0^{hat}
            if self.model.parameterization != 'v':
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                # 모델에서? 이게 ddim 인거 같음
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            
            # FR 이용
            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0)
                warp_D_x0_t = F.grid_sample(D_x0_t, self.grid, align_corners=True)
                residual = self.image_encoder.get_grad


            c1 = a_prev.sqrt() * (1 - a_t)