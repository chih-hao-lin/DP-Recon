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
                 num_views=-1
                 ):

        # self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))
        self.instance_dir = os.path.join(data_root_dir, data_dir, 'scan{0}'.format(scan_id))
        print(self.instance_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
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
            
        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))
        instance_mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "instance_mask", "*.png"))
        
        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
        else:
            mask_paths = None

        self.n_images = len(image_paths)
        print('[INFO]: Dataset Size ', self.n_images)

        # NOTE: we need to fix the camera pose as z up
        # 1. replica scan3 need to fix y axis -7.8 degree
        # 2. youtube scan1 need to fix x axis -10.1 degree
        # 3. youtube scan2 need to fix x axis -7.0 degree
        self.need_fix_x = False
        self.need_fix_y = False
        scan_name = 'scan{0}'.format(scan_id)
        if data_dir == 'replica' and 'scan3' in scan_name:
            self.need_fix_y = True
            self.fix_y_deg = -7.8
            print(f'[INFO]: Fixing camera pose for {data_dir} {scan_name} by rotating along y axis {self.fix_y_deg} degree')
        elif data_dir == 'youtube':
            if 'scan1' in scan_name:
                self.need_fix_x = True
                self.fix_x_deg = -10.1
                print(f'[INFO]: Fixing camera pose for {data_dir} {scan_name} by rotating along x axis {self.fix_x_deg} degree')
            elif 'scan2' in scan_name:
                self.need_fix_x = True
                self.fix_x_deg = -7.0
                print(f'[INFO]: Fixing camera pose for {data_dir} {scan_name} by rotating along x axis {self.fix_x_deg} degree')
            elif 'scan3' in scan_name:
                self.need_fix_x = True
                self.fix_x_deg = -3.0
                print(f'[INFO]: Fixing camera pose for {data_dir} {scan_name} by rotating along x axis {self.fix_x_deg} degree')
            else:
                print(f'[INFO]: No need to fix camera pose for {data_dir} {scan_name}')
        else:
            print(f'[INFO]: No need to fix camera pose for {data_dir} {scan_name}')
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

            if self.need_fix_y:
                pose = rot_cameras_along_y(pose, self.fix_y_deg)         # need to fix the camera pose along y axis for replica scan3
            if self.need_fix_x:
                pose = rot_cameras_along_x(pose, self.fix_x_deg)         # need to fix the camera pose along x axis for youtube scan1 and scan2

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []

        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        # load instance mask and map to label_mapping
        self.semantic_images = []
        self.instance_dilated_region_list = []
        for im_path in instance_mask_paths:
            
            instance_mask_pic = cv2.imread(im_path, -1)
            if len(instance_mask_pic.shape) == 3:
                instance_mask_pic = instance_mask_pic[:, :, 0]
            instance_mask = instance_mask_pic.reshape(1, -1).transpose(1, 0)  # [HW, 1]
            instance_mask[instance_mask==255] = 0         # background is 0

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