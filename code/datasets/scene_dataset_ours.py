import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
import json
from PIL import Image

def rot_cameras_along_x(pose_matrix, angle):
    theta = angle * np.pi / 180
    R = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

    pose_matrix = R @ pose_matrix

    return pose_matrix

def rot_cameras_along_y(pose_matrix, angle):
    theta = angle * np.pi / 180
    R = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

    pose_matrix = R @ pose_matrix

    return pose_matrix

# Dataset with monocular depth, normal and segmentation mask
class SceneDatasetDN_segs(torch.utils.data.Dataset):

    def __init__(self,
                 data_root_dir,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1,
                 split='train',
                 test_split_ratio=0.1,
                 ):

        self.instance_dir = os.path.join(data_root_dir, data_dir, 'room_{0}'.format(scan_id))
        print(self.instance_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        self.need_fix_x = False
        self.need_fix_y = False
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        with open(os.path.join(self.instance_dir, 'instance_id.json'), 'r') as f:
            id_dict = json.load(f)
        f.close()
        self.instance_dict = id_dict
        self.instance_ids = list(self.instance_dict.values())
        self.label_mapping = [0] + self.instance_ids  # background ID is 0 and at the first of label_mapping
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
        image_paths = glob_data(os.path.join(self.instance_dir, "images", "*.jpg"))
        depth_paths = glob_data(os.path.join(self.instance_dir, "depth", "*.npy"))
        normal_paths = glob_data(os.path.join(self.instance_dir, "normal", "*.png"))
        instance_mask_paths = glob_data(os.path.join(self.instance_dir, "instance_mask", "*.png"))
        
        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
        else:
            mask_paths = None

        self.n_images = len(image_paths)
        print('[INFO]: Dataset Size ', self.n_images)
        num_test_split = int(self.n_images * test_split_ratio)
        train_split_indices = np.linspace(0, self.n_images - 1, self.n_images - num_test_split).astype(np.int32)
        test_split_indices = np.setdiff1d(np.arange(self.n_images), train_split_indices).tolist()
        train_split_indices = train_split_indices.tolist()
        split_indices = None
        if split == 'train':
            split_indices = train_split_indices
        elif split == 'test':
            split_indices = test_split_indices
        self.n_images = len(split_indices)

        camera_info_path = os.path.join(self.instance_dir, "transforms.json")
        with open(camera_info_path, 'r') as f:
            camera_info = json.load(f)

        fx = camera_info['fl_x']
        fy = camera_info['fl_y']
        cx = camera_info['cx']
        cy = camera_info['cy']

        intrinsics = np.eye(4)
        intrinsics[:3, :3] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        t_all = []
        for i in range(len(camera_info['frames'])):
            pose = np.array(camera_info['frames'][i]['transform_matrix']).reshape(4, 4)
            t = pose[:3, 3]
            t_all.append(t)
        t_all = np.array(t_all)
        t_max = np.max(t_all, axis=0)
        t_min = np.min(t_all, axis=0)
        norm_scale = np.max(t_max - t_min)
        norm_center = 0.5 * (t_max + t_min)

        self.intrinsics_all = []
        self.pose_all = []
        for i in range(self.n_images):
            idx = split_indices[i]
            pose = np.array(camera_info['frames'][idx]['transform_matrix']).reshape(4, 4)
            t = pose[:3, 3]
            t_norm = (t - norm_center) / norm_scale
            pose[:3, 3] = t_norm
            # opengl to opencv
            pose[:, 1:3] *= -1
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for i in range(self.n_images):
            idx = split_indices[i]
            path = image_paths[idx]
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []

        for i in range(self.n_images):
            idx = split_indices[i]
            dpath = depth_paths[idx]
            npath = normal_paths[idx]
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = Image.open(npath)
            normal = np.array(normal).astype(np.float32) / 255.0
            normal = normal.reshape(-1, 3)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        # load instance mask and map to label_mapping
        self.semantic_images = []
        self.instance_dilated_region_list = []
        for i in range(self.n_images):
            idx = split_indices[i]
            im_path = instance_mask_paths[idx]
            
            instance_mask_pic = cv2.imread(im_path, -1)
            if len(instance_mask_pic.shape) == 3:
                instance_mask_pic = instance_mask_pic[:, :, 0]
            instance_mask = instance_mask_pic.reshape(1, -1).transpose(1, 0)  # [HW, 1]
            background = instance_mask == 255
            instance_mask += 1
            instance_mask[background] = 0       # background is 0

            ins_list = np.unique(instance_mask)
            cur_sems = np.copy(instance_mask)
            for i in ins_list:
                if i not in self.label_mapping:
                    cur_sems[instance_mask == i] = self.label_mapping.index(0)
                else:
                    cur_sems[instance_mask == i] = self.label_mapping.index(i)

            self.semantic_images.append(torch.from_numpy(cur_sems).float())

        # load mask
        self.mask_images = []
        if mask_paths is None:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.mask_images.append(mask)
        else:
            for path in mask_paths:
                mask = np.load(path)
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx],
            "segs": self.semantic_images[idx]
        }

        if self.sampling_idx is not None:
            if (self.random_image_for_path is None) or (idx not in self.random_image_for_path):
                # print('sampling_idx:', self.sampling_idx)
                ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
                ground_truth["full_rgb"] = self.rgb_images[idx]
                ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
                ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
                ground_truth["full_depth"] = self.depth_images[idx]
                ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
                ground_truth["full_mask"] = self.mask_images[idx]
                ground_truth["segs"] = self.semantic_images[idx][self.sampling_idx, :]
            
                sample["uv"] = uv[self.sampling_idx, :]
                sample["is_patch"] = torch.tensor([False])
            else:
                # sampling a patch from the image, this could be used for training with depth total variational loss
                # a fix patch sampling, which require the sampling_size should be a H*H continuous path
                patch_size = np.floor(np.sqrt(len(self.sampling_idx))).astype(np.int32)
                start = np.random.randint(self.img_res[1]-patch_size +1)*self.img_res[0] + np.random.randint(self.img_res[1]-patch_size +1) # the start coordinate
                idx_row = torch.arange(start, start + patch_size)
                patch_sampling_idx = torch.cat([idx_row + self.img_res[1]*m for m in range(patch_size)])
                ground_truth["rgb"] = self.rgb_images[idx][patch_sampling_idx, :]
                ground_truth["full_rgb"] = self.rgb_images[idx]
                ground_truth["normal"] = self.normal_images[idx][patch_sampling_idx, :]
                ground_truth["depth"] = self.depth_images[idx][patch_sampling_idx, :]
                ground_truth["full_depth"] = self.depth_images[idx]
                ground_truth["mask"] = self.mask_images[idx][patch_sampling_idx, :]
                ground_truth["full_mask"] = self.mask_images[idx]
                ground_truth["segs"] = self.semantic_images[idx][patch_sampling_idx, :]
            
                sample["uv"] = uv[patch_sampling_idx, :]
                sample["is_patch"] = torch.tensor([True])
        
        return idx, sample, ground_truth


    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size, sampling_pattern='random'):
        if sampling_size == -1:
            self.sampling_idx = None
            self.random_image_for_path = None
        else:
            if sampling_pattern == 'random':
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
                self.random_image_for_path = None
            elif sampling_pattern == 'patch':
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
                self.random_image_for_path = torch.randperm(self.n_images, )[:int(self.n_images/10)]
            else:
                raise NotImplementedError('the sampling pattern is not implemented.')

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
    
def test_dataset():
    data_root_dir = "/hdd/indoor_digital_twin/DP-Recon/data/"
    data_dir = "replica"
    scan_id = 0
    img_res = (512, 512)

    dataset = SceneDatasetDN_segs(
        data_root_dir=data_root_dir,
        data_dir=data_dir,
        img_res=img_res,
        scan_id=scan_id,
    )

    poses_all = torch.stack(dataset.pose_all).numpy()
    print('poses_all.shape:', poses_all.shape)
    ts = poses_all[:, :3, 3]
    # print(ts)
    max_ts = np.max(ts, axis=0)
    min_ts = np.min(ts, axis=0)
    dist_ts = max_ts - min_ts
    print('dist_ts:', dist_ts)
    center = 0.5 * (max_ts + min_ts)
    print('center:', center)

    idx, sample, ground_truth = dataset[0]
    # print("Sample: ", sample)
    # print("Ground Truth: ", ground_truth)


if __name__ == "__main__":
    test_dataset()