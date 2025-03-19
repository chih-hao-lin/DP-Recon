import os
import numpy as np
import cv2
import json
from PIL import Image
import torch
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from glob import glob
import argparse


@torch.no_grad()
def compute_iou(pred, target):
    """
    Input:
        x: [K, N]
        y: [K, N]
    Return:
        iou: [K, N]
    """
    eps = 1e-8
    pred = pred > 0.5 # [K, N]
    target = target > 0.5
    intersection = (pred & target).sum(-1).float()
    union = (pred | target).sum(-1).float() + eps # [K]
    return (intersection / union).mean()
compute_iou_jit = torch.jit.script(compute_iou)


parser = argparse.ArgumentParser(
    description='Arguments to evaluate total scene reconstruction metrics.'
)
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--root_path', type=str, default='./')
args = parser.parse_args()

dataset_type = args.dataset_type
root_path = args.root_path
exp_root_path = os.path.join(root_path, 'exps')
data_root_path = os.path.join(root_path, f'data/{dataset_type}')

save_iou_txt_path = os.path.join(exp_root_path, f'mask_iou_{dataset_type}.txt')

iou_results = 0                # for all scenes
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

    eval_mask_root_path = os.path.join(exp_path, 'eval_nerf_mask')
    gt_mask_root_path = os.path.join(data_root_path, f'scan{scan_id}', 'eval-novel-views', 'instance_mask')
    eval_mask_list = os.listdir(eval_mask_root_path)
    eval_mask_list = [x for x in eval_mask_list if '.npy' in x]
    eval_mask_list.sort()

    temp_scene_iou = 0
    eval_view_num = 0

    for eval_mask_name in eval_mask_list:
        eval_mask = np.load(os.path.join(eval_mask_root_path, eval_mask_name)).astype(np.uint8)

        view_id = int(eval_mask_name.split('_')[-1].split('.')[0])
        gt_mask_name = f'{view_id:06d}.png'
        gt_mask_path = os.path.join(gt_mask_root_path, gt_mask_name)
        gt_mask = cv2.imread(gt_mask_path, -1)
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[:, :, 0]
        gt_mask[gt_mask==255] = 0                       # background is 0
        # map to object id in network (0, 1, 2 ...)
        instance_json = os.path.join(data_root_path, f'scan{scan_id}', 'instance_id.json')
        with open(instance_json, 'r') as f:
            instance_dict = json.load(f)
        instance_ids = list(instance_dict.values())
        label_mapping = [0] + instance_ids  # background ID is 0 and at the first of label_mapping
        ins_list = np.unique(gt_mask)
        for i in ins_list:
            gt_mask[gt_mask==i] = label_mapping.index(i)

        # eval mask
        h, w = gt_mask.shape
        K = len(np.unique(gt_mask))

        if K == 1:                          # only background
            continue

        gt, pred = torch.tensor(gt_mask).cuda(), torch.tensor(eval_mask).cuda()
        gt_onehot, pred_onehot = torch.zeros(K, h, w).cuda(), torch.zeros(K, h, w).cuda()
        for i in range(K):
            id = torch.unique(gt)[i]
            gt_onehot[i] = (gt == id).float()
            pred_onehot[i] = (pred == id).float()
        gt_onehot, pred_onehot = gt_onehot[1:].reshape(K-1, h*w), pred_onehot[1:].reshape(K-1, h*w)

        iou = compute_iou(pred_onehot.float(), gt_onehot.float())
        temp_scene_iou += iou.item()
        eval_view_num += 1

    scene_iou = temp_scene_iou / eval_view_num
    iou_results += scene_iou
    total_scene += 1

    with open(save_iou_txt_path, 'a') as f:
        out = f"scan{scan_id} mask IoU: {scene_iou}\n"
        f.write(out)

    print(f'scan{scan_id} mask IoU: {scene_iou}')

mean_iou = iou_results / total_scene
with open(save_iou_txt_path, 'a') as f:

    out = f'dataset: {dataset_type}, total scene: {total_scene}\n'
    f.write(out)

    out = f"mean mask IoU: {mean_iou}\n"
    f.write(out)

print(f'dataset: {dataset_type}, total scene: {total_scene}')
print(f'mean mask IoU: {mean_iou}')
