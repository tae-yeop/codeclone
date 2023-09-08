




device = 'cuda'
parser = argparse.ArgumentParser()
...
opt = parser.parse_args()

save_video_frames()

prep(opt)



import torchvision.transforms as T
from torchvision.io import read_video, write_video

def save_video_frames(video_path, img_size=(512, 512)):
    video, _, _ = read_video(video_path, output_format='TCHW')

    if video_path.endswith('.mov'):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem

    os.makedirs(f'data/{video_name}', exists_ok=True)

    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])

def save_video(raw_frames, save_path, fps=10):
    video_codec = ''
    video_options = ...
    # 채널을 뒤로 보냄
    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    # write는 [T, H, W, C]를 저장함 : https://pytorch.org/vision/main/generated/torchvision.io.write_video.html
    # read할때는 [T,H,W,C] 혹은 [T, C, H, W]로 할 수 있다. https://pytorch.org/vision/main/generated/torchvision.io.read_video.html
    # 프레임을 어떻게 해서 저장할지 
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)
    


def get_timesteps(scheduler, num_inference_steps, strength, device):
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheudler.time


def prep(opt):
    model_key = "stabilityai/stable-diffusion-2-1-base"

    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder='scheduler')
    toy_scheduler.set_timesteps(opt.save_steps)
    timesteps_to_save, num_inference_steps = get_timesteps()

    seed_everything(1)


    model = Preprocess(device, opt)

    # inversion하면서 recon
    recon_frames = model.extract_latents(num_steps=opt.steps, # timestep 설정
                                         save_path=save_path,
                                         batch_size=opt.batch_size,
                                         timesteps_to_save=timesteps_to_save,
                                         inversion_prompt=opt.inversion_prompt)


    if not os.path.isdir(os.path.join(save_path, f'frames')):
        os.mkdir(os.path.join(save_path, f'frames'))

    # PIL 이미지로 저장
    for i, frame in enumerate(recon_frames):
        T.ToPILImage()(frame).save(os.path.join(save_path, f'frames', f'{i:05d}.png'))
    frames = (recon_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(os.path.join(save_path, f'inverter.mp4'), frames, fps=10)w