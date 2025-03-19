# copy from https://huggingface.co/docs/diffusers/conceptual/evaluation
import torch
import numpy as np
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")

def calculate_clip_score(images, prompts):
    '''
    images: np.ndarray, shape (B, H, W, C), range [0, 1]
    '''
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

prompts = [
    "a chair", 
    "a table"
]

# test 1
chair_path_1 = '/mnt/fillipo/Datasets/Pix3d/pix3d_full/img/chair/0016.png'
table_path_1 = '/mnt/fillipo/Datasets/Pix3d/pix3d_full/img/table/0001.png'
H, W = 384, 384
chair_img_1 = Image.open(chair_path_1)
table_img_1 = Image.open(table_path_1)
chair_img_1 = chair_img_1.resize((W, H))
table_img_1 = table_img_1.resize((W, H))
chair_img_1 = np.array(chair_img_1) / 255.0
table_img_1 = np.array(table_img_1) / 255.0
print(chair_img_1.shape, table_img_1.shape)
images1 = np.stack([chair_img_1, table_img_1], axis=0)

sd_clip_score = calculate_clip_score(images1, prompts)
print(f"images1 CLIP score: {sd_clip_score}")

# test 2
chair_path_2 = '/mnt/fillipo/Datasets/Pix3d/pix3d_full/img/table/0001.png'
table_path_2 = '/mnt/fillipo/Datasets/Pix3d/pix3d_full/img/chair/0016.png'
H, W = 384, 384
chair_img_2 = Image.open(chair_path_2)
table_img_2 = Image.open(table_path_2)
chair_img_2 = chair_img_2.resize((W, H))
table_img_2 = table_img_2.resize((W, H))
chair_img_2 = np.array(chair_img_2) / 255.0
table_img_2 = np.array(table_img_2) / 255.0
print(chair_img_2.shape, table_img_2.shape)
images2 = np.stack([chair_img_2, table_img_2], axis=0)

sd_clip_score = calculate_clip_score(images2, prompts)
print(f"images2 CLIP score: {sd_clip_score}")
