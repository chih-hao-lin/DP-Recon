import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO
import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_image_from_path(img_path):
    with open(img_path, 'rb') as f:
        image_bytes = f.read()
    image = Image.open(BytesIO(image_bytes))
    return image, image_bytes


def get_scene_musiq(eval_rgb_root_path, predict_fn):
    eval_rgb_list = os.listdir(eval_rgb_root_path)
    eval_rgb_list.sort()

    temp_scene_musiq = 0

    for filename in eval_rgb_list:

        img_path = os.path.join(eval_rgb_root_path, filename)
        image, image_bytes = load_image_from_path(img_path)

        prediction = predict_fn(tf.constant(image_bytes))
        
        musiq_score = prediction['output_0'].numpy().item()
        temp_scene_musiq += musiq_score

    scene_musiq = temp_scene_musiq / len(eval_rgb_list)
    return scene_musiq


NAME_TO_HANDLE = {
    # Model trained on SPAQ dataset: https://github.com/h4nwei/SPAQ
    'spaq': 'https://tfhub.dev/google/musiq/spaq/1',

    # Model trained on KonIQ-10K dataset: http://database.mmsp-kn.de/koniq-10k-database.html
    'koniq': 'https://tfhub.dev/google/musiq/koniq-10k/1',

    # Model trained on PaQ2PiQ dataset: https://github.com/baidut/PaQ-2-PiQ
    'paq2piq': 'https://tfhub.dev/google/musiq/paq2piq/1',

    # Model trained on AVA dataset: https://ieeexplore.ieee.org/document/6247954
    'ava': 'https://tfhub.dev/google/musiq/ava/1',
}


parser = argparse.ArgumentParser(
    description='Arguments to evaluate total scene reconstruction metrics.'
)
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--root_path', type=str, default='./')
args = parser.parse_args()

dataset_type = args.dataset_type
root_path = args.root_path
exp_root_path = os.path.join(root_path, 'exps')

save_musiq_txt_path = os.path.join(exp_root_path, f'musiq_{dataset_type}.txt')

selected_model = 'koniq' #@param ['spaq', 'koniq', 'paq2piq', 'ava']
model_handle = NAME_TO_HANDLE[selected_model]
model = hub.load(model_handle)
predict_fn = model.signatures['serving_default']
print(f'loaded model {selected_model} ({model_handle})')

musiq_results = {}
total_scene = 0

exp_list = os.listdir(exp_root_path)
exp_list.sort()
for exp_name in exp_list:
    if f'_{dataset_type}_' not in exp_name:
        continue

    scan_id = exp_name.split('_')[-1]
    temp_exp_path = os.path.join(exp_root_path, exp_name)
    run_exp_list = os.listdir(temp_exp_path)
    run_exp_list.sort()
    run_exp_name = run_exp_list[-1]             # use the latest exp
    exp_path = os.path.join(temp_exp_path, run_exp_name, 'plots')

    eval_nerf_rgb_root_path = os.path.join(exp_path, 'eval_nerf_rgb')
    nerf_scene_musiq = get_scene_musiq(eval_nerf_rgb_root_path, predict_fn)
    if 'nerf_rgb_musiq' not in musiq_results:
        musiq_results['nerf_rgb_musiq'] = nerf_scene_musiq
    else:
        musiq_results['nerf_rgb_musiq'] += nerf_scene_musiq

    eval_nvdiff_rgb_root_path = os.path.join(exp_path, 'eval_nvdiffrast_rgb')
    nvdiff_scene_musiq = get_scene_musiq(eval_nvdiff_rgb_root_path, predict_fn)
    if 'nvdiff_rgb_musiq' not in musiq_results:
        musiq_results['nvdiff_rgb_musiq'] = nvdiff_scene_musiq
    else:
        musiq_results['nvdiff_rgb_musiq'] += nvdiff_scene_musiq

    total_scene += 1


print(f'dataset: {dataset_type}, total scene: {total_scene}')
for k, v in musiq_results.items():
    print(f'{k}: {v/total_scene}')

with open(save_musiq_txt_path, 'w') as f:
    f.write(f'dataset: {dataset_type}, total scene: {total_scene}\n')
    for k, v in musiq_results.items():
        f.write(f'{k}: {v/total_scene}\n')


