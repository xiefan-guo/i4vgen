import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision
from torchvision.utils import save_image

from diffusers import AutoencoderKL, DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from i4vgen.animatediff.models.unet import UNet3DConditionModel
from i4vgen.animatediff.pipelines.pipeline_animation import AnimationPipeline
from i4vgen.animatediff.utils.util import save_videos_grid
from i4vgen.animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange
from pathlib import Path

import ImageReward as RM


def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir, exist_ok=True)

    config  = OmegaConf.load(args.config)

    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

            ### >>> create validation pipeline >>> ###
            tokenizer           = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder        = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae                 = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet                = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
            image_reward_model  = RM.load("ImageReward-v1.0")

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                image_reward_model=image_reward_model, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            pipeline = load_weights(
                pipeline,
                # motion module
                motion_module_path         = motion_module,
                motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
                # image layers
                dreambooth_model_path      = model_config.get("dreambooth_path", ""),
                lora_model_path            = model_config.get("lora_model_path", ""),
                lora_alpha                 = model_config.get("lora_alpha", 0.8),
            ).to("cuda")

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

            config[config_key].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())

                print(f"current seed: {torch.initial_seed()}")
                print(f"Processing the ({prompt}) prompt")

                # Save intermediate results 
                sample = pipeline(
                    prompt,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = args.W,
                    height              = args.H,
                    video_length        = args.L,
                )

                sample, candidate_images, ni_vsds_video = sample.videos, sample.candidate_images, sample.ni_vsds_video
                
                video_name = f"{savedir}/{prompt}-{random_seed}.mp4"
                sample_mp4 = sample[0].contiguous().permute(1, 2, 3, 0).contiguous()
                sample_mp4 = (sample_mp4 * 255)
                sample_mp4 = sample_mp4.to(dtype=torch.uint8)
                torchvision.io.write_video(video_name, sample_mp4, fps=8)

                '''candidate_images'''
                # sample_png = rearrange(candidate_images, "b c f h w -> b c h (f w)").contiguous()
                # png_name = f"{savedir}/{prompt}-{random_seed}-candidate-images.png"
                # save_image(sample_png, png_name)

                '''ni_vsds_video'''
                # video_name = f"{savedir}/{prompt}-{random_seed}-ni-vsds-video.mp4"
                # sample_mp4 = ni_vsds_video[0].contiguous().permute(1, 2, 3, 0).contiguous()
                # sample_mp4 = (sample_mp4 * 255)
                # sample_mp4 = sample_mp4.to(dtype=torch.uint8)
                # torchvision.io.write_video(video_name, sample_mp4, fps=8)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="[path to base T2I diffusion model]",)
    parser.add_argument("--inference_config",      type=str, default="configs/animatediff_configs/inference/inference-v2.yaml")    
    parser.add_argument("--config",                type=str, required=True)

    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
