from pipeline import DragPipeline


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = rearrage(image, "h  c -> 1 c h w")
    image = image.to(device)
    return image


def gen_imag(
    length,
    height,
    width,
    n_inference_step,
    scheduler_name,
    seed,
    guidance_scale,
    prompt,
    neg_prompt,
    model_path,
    vae_path,
    lora_path
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = 
def run_drag(source_image,
             image_with_clicks,
             mask,
             prompt,
             inversion_strength,
             lam,
             latent_lr,
             n_pix_step,
             model_path,
             vae_path,
             lora_path,
             ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    scheduler = DDIMScheduler()
    model = DragPipeline.from_pretrained(model_path, 
                                         scheduler=scheduler,
                                         cache_dir='').to(device)
    model.modify_unet_forward()


    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(vae_path).to(model.vae.device, model.vae.dtype)

    seed = 42
    seed_everything(seed)


    args = SimpleNamespace()
    args.prompt = prompt
    args.neg_prompt = neg_prompt
    args.points = points
    args.n_inference_step = n_inference_step
    args.n_actual_inference_step = round(n_inference_step * inversion_strength)
    args.guidance_scale = guidance_scale

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)

    if lora_path != '':
        model.load_lora_weights(lora_path, weight_name='lora.safetensors')

    mask = torch.from_numpy

    
    gen_image = model(prompt=args.prompt,
                      batch_size=2,
                      la)


def run_drag_gen(
    n_inference_step,
    scheduler_name,
    source_image,
    image_with_clicks,
    intermediate_latents_gen,
    
    
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DragPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    if schduler_name == "DDIM":
        scheduler = DDIMScheduler()
    elif scheduler_name == "DPM++2M":
        scheduler = DPMSolverMultistepScheduler.from_config(
            model.scheduler.config
        )
    elif scheduler_name == "DPM++2M_karras":
        scheduler = DPMSolverMultistepScheduler.from_config(
            model.scheduler.config, use_karras_sigmas=True
        )
    else:
        raise NotImplementedError("scheduler name not correct")

    model.scheduler = scheduler
    model.modify_unet_forwrad()


    if vae_path != 'default':
        model.vae = AutoencoderKL.from_pretrained(vae_path).to(model.vae.device, model.vae.dtype)


    seed = 42
    seed_everything(seed)


    args = SimpleNamespace()
    args.prompt = prompt
    args.neg_prompt = neg_prompt
    args.points = points
    args.n_inference_step = n_inference_step
    args.n_actual_inference_step = round(n_inference_step * inversion_strength)
    args.guidance_scale = guidance_scale

    args.unet_feature_idx = [3]

    full_h, full_w = source_image.shape[:2]

    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = latent_lr

    args.n_pix_step = n_pix_step
    print(args)

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)

    if lora_path != "":
        print("applying lora: " + lora_path)
        model.load_lora_weights(lora_path, weight_name="lora.safetensors")

    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")

    
    handle_points = []
    target_points = []

    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)

    print('handle points:', handle_points)
    print('target points:', target_points)

    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]
    init_code = deepcopy(intermediate_latents_gen[args.n_inference_step - args.n_actual_inference_step])
    init_code_orig = deepcopy(init_code)
    
    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    init_code.to(torch.float32)
    model = model.to(device, torch.float32)
    updated_init_code = drag_diffusion_update_gen(model, init_code, t, handle_points, target_points,
                                                  mask, args)
    updated_init_code = updated_init_code.to(torch.float16)
    model = model.to(device, torch.float16)

    editor = MutualSelfAttentionControl()

    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    gen_image = model(prompt=args.prompt,
                      neg_prompt=args.neg_prompt,
                      batch_size=2,
                      latents=torch.cat([init_code_orig, updated_init_code], dim=0),
                      guidance_scale=args.guidance_scale,
                      num_inference_steps=arg.n_inference_step,
                      num_actual_inference_steps=args.n_actual_inference_step)[1].unsqueeze(dim=0)

    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

    save_result = torch.cat([soruce_image * 0.5 + 0.5,
                             torch.ones((1,3,full_h,25)).cuda(),
                             image_with_clicks * 0.5 + 0.5,
                             torch.ones((1,3,full_h,25)).cuda(),
                             gen_image[0:1]], dim=-1)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_image()

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = 