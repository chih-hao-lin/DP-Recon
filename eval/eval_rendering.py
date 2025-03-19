import os
import cv2
import torch
import argparse
import torch.nn as nn
from glob import glob
from piq import ssim 
from piq import psnr 
import lpips


class ReconMetrics(nn.Module):
    def __init__(self, lpips_net='vgg', batchsize=8):
        super().__init__()
        self.batchsize = batchsize
        self.metrics = {
            "ssim": ssim,
            "psnr": psnr,
            "lpips": lpips.LPIPS(net=lpips_net),
        }

    def forward(self, preds, targets):
        r"""
        Input:
            preds: [B, C, H, W]
            targets: [B, C, H, W]
        Return:
            metrics: dict of metrics
        """
        metrics = {}
        for k, v in self.metrics.items():
            if len(preds) < self.batchsize:
                metrics[k] = v(preds, targets).mean().item()
            else:
                metric = []
                for i in range(0, len(preds)//self.batchsize + 1):
                    pred = preds[i*self.batchsize:(i+1)*self.batchsize]
                    target = targets[i*self.batchsize:(i+1)*self.batchsize]
                    if k != 'lpips':
                        eval_batch = v(pred, target, reduction='none')
                    else:
                        eval_batch = v(pred, target)

                    metric.append(eval_batch.mean().item() * len(pred))
                metrics[k] = sum(metric) / len(preds)

        return metrics
        
    def compute(self, preds, targets, metric='psnr'):
        return self.metrics[metric](preds, targets)
    
    def metrics_name(self):
        return list(self.metrics.keys())
    
    def set_divice(self, device):
        self.metrics["lpips"].to(device)


def load_images(pred_dir, target_dir):
    postfix = 'png'
    pred_paths = sorted(glob(os.path.join(pred_dir, f'*.{postfix}')), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    target_paths = sorted(glob(os.path.join(target_dir, f'*_rgb.{postfix}')), key=lambda x: int(x.split('/')[-1].split('_')[0]))
    assert len(pred_paths) > 0 and len(target_paths) > 0, f'no images found in {pred_dir} or {target_dir}'
    assert len(pred_paths) == len(target_paths), f'number of images in {pred_dir} and {target_dir} are not equal'

    pred_images = []
    for f in pred_paths:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        pred_images.append(img)
    target_images = []
    for f in target_paths:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        target_images.append(img)
    return torch.cat(pred_images, dim=0).cuda(), torch.cat(target_images, dim=0).cuda() # [B, C, H, W]


def eval_rendering(pred_dir, target_dir, lpips_net='vgg', batchsize=16):
    recon_metrics = ReconMetrics(lpips_net, batchsize)
    preds, targets = load_images(pred_dir, target_dir)
    recon_metrics.set_divice(preds.device)
    metrics = recon_metrics(preds, targets)
    return metrics


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

nerf_results = {}                       # for all scenes
nvdiffrast_results = {}
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

    target_dir = os.path.join(data_root_path, f'scan{scan_id}', 'eval-novel-views', 'rgb')

    # 1.eval nerf rgb
    test_name = 'eval_nerf_rgb'
    pred_dir = os.path.join(exp_path, test_name)
    nerf_metrics = eval_rendering(pred_dir, target_dir)

    for k, v in nerf_metrics.items():
        if k not in nerf_results:
            nerf_results[k] = v
        else:
            nerf_results[k] += v

    # 2.eval nvdiffrast rgb
    test_name = 'eval_nvdiffrast_rgb'
    pred_dir = os.path.join(exp_path, test_name)
    nvdiffrast_metrics = eval_rendering(pred_dir, target_dir)

    for k, v in nvdiffrast_metrics.items():
        if k not in nvdiffrast_results:
            nvdiffrast_results[k] = v
        else:
            nvdiffrast_results[k] += v

    total_scene += 1

print(f'dataset: {dataset_type}, total scene: {total_scene}')

print('nerf rendering results:')
for k, v in nerf_results.items():
    print(f'nerf {k}: {v/total_scene}')
total_nerf_results_txt_path = os.path.join(exp_root_path, f'nerf_rendering_{dataset_type}.txt')
with open(total_nerf_results_txt_path, 'w') as f:
    for k, v in nerf_results.items():
        f.write(f'{k}: {v/total_scene}\n')

print('nvdiffrast rendering results:')
for k, v in nvdiffrast_results.items():
    print(f'nvdiffrast {k}: {v/total_scene}')
total_nvdiffrast_results_txt_path = os.path.join(exp_root_path, f'nvdiffrast_rendering_{dataset_type}.txt')
with open(total_nvdiffrast_results_txt_path, 'w') as f:
    for k, v in nvdiffrast_results.items():
        f.write(f'{k}: {v/total_scene}\n')


