from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image

import cv2
from PIL import Image
import numpy as np
import torch
import os
import argparse
import shutil
import random


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--inpaint_repeat_times', type=int, required=True)
args = parser.parse_args()

root_dir = args.root_dir
save_root_path = args.save_dir
inpaint_repeat_times = args.inpaint_repeat_times
os.makedirs(save_root_path, exist_ok=True)

input_path = os.path.join(root_dir, f'rgb_map.png')
depth_path = os.path.join(root_dir, f'disp.png')
mask_path = os.path.join(root_dir, f'vis_mask.png')

save_input_path = os.path.join(save_root_path, 'rgb_map.png')
shutil.copy(input_path, save_input_path)

init_image = Image.open(input_path).convert("RGB")
init_img_arr = np.array(init_image)
init_mask = Image.open(mask_path).convert("RGB")
depth_image = Image.open(depth_path).convert("RGB")
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)

iter_num_list = [3, 6, 9]
dilated_mask_dict = {}
for iter_num in iter_num_list:
    mask_obj_arr = np.array(init_mask)
    for i in range(iter_num):
        mask_obj_arr = cv2.dilate(mask_obj_arr, kernel, 3)

    dilated_mask = Image.fromarray(mask_obj_arr)
    dilated_mask_dict[iter_num] = dilated_mask
    masked_img_arr = init_img_arr
    masked_img_arr[np.where(mask_obj_arr == 255)] = 255
    masked_img = Image.fromarray(masked_img_arr)
    masked_img.save(f'{save_root_path}/{iter_num}_masked.png')

    # save mask map
    dilated_mask.save(f'{save_root_path}/{iter_num}_vis_mask.png')

# Load the model
controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

pipe.to('cuda')
text_prompt = 'panoramic image, empty room, award-winning, professional, highly detailed, nothing on the floor and wall'
negative_prompt = 'anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry, low quality, worst quality, inconsistent, rough surface'

for iter_num in iter_num_list:
    dilated_mask = dilated_mask_dict[iter_num]
    for strength in range(4, 7):
        for num_id in range(inpaint_repeat_times):
            new_image = pipe(
                text_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                image=init_image,
                control_image=depth_image,
                mask_image=dilated_mask,
                guidance_scale=7.5,
                controlnet_conditioning_scale=strength / 10,
                width=2048,
                height=1024
            ).images[0]
            new_image.save(f'{save_root_path}/iter_num_{iter_num}_out_{strength:02d}_{num_id:03d}.png')
            print(f'iter_num_{iter_num}_{strength:02d}_{num_id:03d} done')
print('finish pano inpainting!')

# randomly select one result
save_bg_inpaint_path = os.path.join(root_dir, 'bg_inpaint.png')
inpaint_result_list = os.listdir(save_root_path)
inpaint_result_list = [result for result in inpaint_result_list if result.endswith('.png')]
random_inpaint_result_name = random.choice(inpaint_result_list)
random_inpaint_result_path = os.path.join(save_root_path, random_inpaint_result_name)
shutil.copy(random_inpaint_result_path, save_bg_inpaint_path)
print('[INFO]: finish inpaint bg panorama. A result has been randomly selected, but manually choosing the best one from all results would yield better quality.')
print(f'[INFO]: all the inpainting results are saved at {save_root_path}, please choose the best one to replace {save_bg_inpaint_path}')
print('********** finish inpaint bg panorama, end decompositional reconstruction and add geometry prior stage, exit **********')
