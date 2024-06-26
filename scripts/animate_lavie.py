import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import argparse
import torchvision

from i4vgen.lavie.pipelines.pipeline_videogen import VideoGenPipeline

from i4vgen.lavie.utils.download import find_model
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from omegaconf import OmegaConf

from i4vgen.lavie.models import get_models
import imageio

from einops import rearrange
from torchvision.utils import save_image
import ImageReward as RM


def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
	
    sd_path = 'CompVis/stable-diffusion-v1-4'
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device) # huge
    image_reward_model  = RM.load("ImageReward-v1.0")

    # set eval mode
    unet.eval()
    vae.eval()
    text_encoder_one.eval()

    if args.sample_method == 'ddim':
        scheduler = DDIMScheduler.from_pretrained(sd_path, 
			subfolder="scheduler",
			beta_start=args.beta_start, 
			beta_end=args.beta_end, 
			beta_schedule=args.beta_schedule)
    elif args.sample_method == 'eulerdiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(sd_path,
			subfolder="scheduler",
			beta_start=args.beta_start,
			beta_end=args.beta_end,
			beta_schedule=args.beta_schedule)
    elif args.sample_method == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(sd_path,
			subfolder="scheduler",
			beta_start=args.beta_start,
			beta_end=args.beta_end,
			beta_schedule=args.beta_schedule)
    else:
        raise NotImplementedError
    
    videogen_pipeline = VideoGenPipeline(
        vae=vae, 
		text_encoder=text_encoder_one, 
		tokenizer=tokenizer_one, 
		scheduler=scheduler, 
		unet=unet).to(device)
    videogen_pipeline.image_reward_model = image_reward_model
    videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for i, cur_seed in enumerate(args.seed):

        torch.manual_seed(cur_seed)
        for prompt in args.text_prompt:
            print('Processing the ({}) prompt'.format(prompt))
            sample = videogen_pipeline(
                prompt=prompt, 
                video_length=args.video_length, 
                height=args.image_size[0], 
                width=args.image_size[1], 
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale)
            
            videos = sample.video
            video_mp4_name = args.output_folder + prompt.replace(' ', ' ') + '-{}.mp4'.format(cur_seed)
            video_mp4 = videos[0]
            torchvision.io.write_video(video_mp4_name, video_mp4, fps=8)

            '''candidate_images'''
            # candidate_images = sample.candidate_images
            # video_png = rearrange(candidate_images, "b f h w c -> (b f) c h w").contiguous()
            # video_png = video_png.float()
            # video_png = video_png / 255.
            # png_name = args.output_folder + prompt.replace(' ', ' ') + '-{}-candidate-images.jpg'.format(cur_seed)
            # save_image(video_png, png_name, nrow=4)

            '''ni_vsds_video'''
            # ni_vsds_video = sample.ni_vsds_video
            # video_mp4_name = args.output_folder + prompt.replace(' ', ' ') + '-{}-ni-vsds-video.mp4'.format(cur_seed)
            # video_mp4 = ni_vsds_video[0]
            # torchvision.io.write_video(video_mp4_name, video_mp4, fps=8)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="")
	args = parser.parse_args()

	main(OmegaConf.load(args.config))
