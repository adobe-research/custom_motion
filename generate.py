# Copyright 2024 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.


import argparse
import torch
import os
from einops import rearrange
from compel import Compel
from train import export_to_video
import numpy as np
import shutil
import uuid
import warnings
import torch
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline
from models.unet_3d_condition import UNet3DConditionModel
from einops import rearrange
from compel import Compel
from train import handle_memory_attention, load_primary_models


def initialize_pipeline(model, device="cuda", xformers=False, sdp=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, _unet = load_primary_models(model)
        del _unet  # This is a no op
        unet = UNet3DConditionModel.from_pretrained(model, subfolder='unet')

    pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    unet._set_gradient_checkpointing(value=False)
    handle_memory_attention(xformers, sdp, unet)
    vae.enable_slicing()
    return pipeline


seed = 100

np.random.seed(seed)

def generate_test(model, prompt, output_dir, seed, negative_prompt=None, res=384, duration=16, steps=50):

    if seed == 'random':
        seed = [np.random.randint(0, 100000) for i in range(10000)]
    else:
        seed = int(seed)
    
    if prompt.endswith('.txt'):
        with open(prompt, 'r') as f:
            all_prompts = f.readlines()
        all_prompts = [p[:-1].split(',') for p in all_prompts]
        prompt_set = os.path.basename(prompt).split('.')[0]
    else:
        all_prompts = prompt.split('|')
        prompt_set = prompt.replace(' ', '_')
    if output_dir is None:
        output_dir = f'{model}/{prompt_set}'
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = initialize_pipeline(model)
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

    for p in all_prompts:
        generator = torch.Generator(device='cuda').manual_seed(seed)
        if type(p) == str:
            pass
        else:
            seed, p = int(p[0]), p[1]
        prompt = compel_proc([p])
        video = pipeline(
            prompt_embeds=prompt,
            negative_prompt_embeds=negative_prompt,
            num_frames=duration,
            width=res,
            height=res,
            num_inference_steps=steps,
            guidance_scale=9,
            output_type="pt",
            generator=generator,
        ).frames
        video = rearrange(video[0], "f c h w -> f h w c")
        video = video.clamp(0, 1).mul(255)
        video = video.byte().cpu().numpy()
        vid_name = f"{os.path.join(output_dir, p.replace(' ', '_'))}_seed{seed}.mp4"
        rnd = str(uuid.uuid4())
        export_to_video(video, f'tmp/{rnd}.mp4', 8)
        command = f'ffmpeg -y -i "tmp/{rnd}.mp4" -c:v libx264 "{vid_name}"'
        os.system(command)
        os.system(f'rm tmp/{rnd}.mp4')

   
    shutil.copy('utils/video_lightbox.html', output_dir)



if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='models/model_scope_diffusers')
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-n", "--negative-prompt", type=str, default=None)
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("-s", "--seed", type=str, default=44)
    parser.add_argument("-r", "--res", type=int, default=384)
    parser.add_argument("-d", "--duration", type=int, default=16)
    parser.add_argument("-steps", type=int, default=50)


    args = parser.parse_args()
    if args.model == 'zs':
        args.model = 'cerspense/zeroscope_v2_576w'
    generate_test(args.model, args.prompt, args.output_dir, args.seed, args.negative_prompt, args.res, args.duration, args.steps)
