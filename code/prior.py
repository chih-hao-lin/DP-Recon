import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

import threestudio
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.base import (Updateable, update_end_if_possible,
                                    update_if_possible,)
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import C, cleanup, get_device, load_module_weights
from threestudio.utils.saving import SaverMixin
from threestudio.utils.typing import *
from threestudio.models.mesh import Mesh

from utils.plots import get_surface_sliding

from omegaconf import OmegaConf
import bisect
import trimesh
from datetime import datetime
import random
import numpy as np
import json
import shutil

from camera_views import CameraViews

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image


class PriorModule(nn.Module, SaverMixin):
    def __init__(self, prior_yaml, recon_sdf, recon_color, recon_density, prior_obj_idx_list, prompt_path, plots_dir, device='cuda'):

        super(PriorModule, self).__init__()

        cfg = OmegaConf.load(prior_yaml)

        self.cfg = cfg
        self.cfg.texture = False
        self.device = device

        if 'texture' in prior_yaml:         # texture prior stage
            self.color_vis_thresh = cfg.system.guidance.color_vis_thresh
            self.bg_vis_thresh = cfg.system.guidance.bg_vis_thresh
        else:
            self.color_vis_thresh = 0.5
            self.bg_vis_thresh = 0.1

        self._root_save_dir = os.path.join(plots_dir, 'sds_views')
        os.makedirs(self._root_save_dir, exist_ok=True)
        save_cfg_path = os.path.join(self._root_save_dir, 'prior.yaml')
        shutil.copyfile(prior_yaml, save_cfg_path)
        self._exp_root_save_dir = os.path.join(self._root_save_dir, 'save')
        self._save_dir = None

        self.true_global_step = 0
        self.global_step = 0

        self.camera_module = CameraViews(cfg.data)

        self.geometry = threestudio.find(cfg.system.geometry_type)(cfg.system.geometry).to(device)
        self.geometry.finite_difference_normal_eps = 0.01
        shape_init_params = cfg.system.geometry.shape_init_params
        print(f'*********** shape_init_params: {shape_init_params} ***********')

        self.geometry.set_sdf_network(recon_sdf, shape_init_params)
        self.geometry.set_color_network(recon_color, shape_init_params)

        self.material = threestudio.find(cfg.system.material_type)(cfg.system.material).to(device)
        
        background_cfg = {}         # follow RichDreamer the first geometry stage
        self.background = threestudio.find(cfg.system.background_type)(background_cfg).to(device)
        
        self.renderer = threestudio.find(cfg.system.renderer_type)(
            cfg.system.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
            recon_density=recon_density,
        ).to(device)

        # init estimator for each object
        self.obj_estimator_dict = {}
        for obj_idx in prior_obj_idx_list:
            self.obj_estimator_dict[obj_idx] = self.renderer.create_estimator().to(device)

        # load prompt
        self.obj_prompt_dict = {}
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)

        self.use_sd_prior = self.cfg.system.use_sd_prior

        self.begin_color_prior = False
        self.color_mesh_dict = {}

        if self.use_sd_prior:
            # Stable Diffusion (SD)
            self.guidance = threestudio.find(self.cfg.system.guidance_type)(self.cfg.system.guidance)

        for obj_idx in prior_obj_idx_list:

            print(f'*********** init prompt for object {obj_idx} ***********')
            obj_prompt = prompts[str(obj_idx)]
            prompt_utils = self.init_prompt(obj_prompt)
            self.obj_prompt_dict[obj_idx] = prompt_utils

    def set_prompt_renderer_idx(self, obj_idx, group_prior_obj_list):

        # set obj_idx
        group_prior_obj_list = torch.tensor(group_prior_obj_list).to(self.device)
        self.geometry.set_obj_idx(group_prior_obj_list)

        # set prompt
        self.prompt_utils = self.obj_prompt_dict[obj_idx]

        # set renderer
        self.renderer.init_estimator(self.obj_estimator_dict[obj_idx])

    def init_prompt(self, obj_prompt):
        ###### NOTE: set for each object, because prompt is object specific

        self.cfg.system.prompt_processor.prompt = obj_prompt

        # Stable Diffusion (SD)
        prompt_processor = threestudio.find(self.cfg.system.prompt_processor_type)(
                            self.cfg.system.prompt_processor
                        )
        prompt_utils = prompt_processor()

        return prompt_utils
    
    def init_color_mesh(self, mesh_path_list, obj_idx, prior_bbox_path):

        mesh_list = []
        for mesh_path in mesh_path_list:
            mesh = trimesh.load(mesh_path)
            mesh_list.append(mesh)
        merge_mesh = trimesh.util.concatenate(mesh_list)

        if prior_bbox_path is not None:
            # normalize obj mesh
            print(f'*********** normalize obj mesh {obj_idx} ***********')
            with open(prior_bbox_path, 'r') as f:
                obj_bbox = json.load(f)

            x_min = obj_bbox[0][0] + 0.1
            y_min = obj_bbox[0][1] + 0.1
            z_min = obj_bbox[0][2] + 0.1
            x_max = obj_bbox[1][0] - 0.1
            y_max = obj_bbox[1][1] - 0.1
            z_max = obj_bbox[1][2] - 0.1
            x_len = x_max - x_min
            y_len = y_max - y_min
            z_len = z_max - z_min
            centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            centroid = np.array(centroid)

            max_len = max(x_len, y_len, z_len)
            shape_scale = self.cfg.system.geometry.shape_init_params / max_len

            merge_mesh.vertices = merge_mesh.vertices - centroid
            merge_mesh.vertices = merge_mesh.vertices * shape_scale                  # in [-0.5, 0.5], if shape_init_params = 0.5
            merge_mesh.vertices = merge_mesh.vertices * 2.0                          # in [-1, 1], if shape_init_params = 0.5

        v_pos = torch.from_numpy(merge_mesh.vertices).float().to(self.device)
        t_pos_idx = torch.from_numpy(merge_mesh.faces).long().to(self.device)

        self.color_mesh_dict[obj_idx] = Mesh(v_pos, t_pos_idx)                       # mesh in threestudio type
    
    def init_color_render_module(self, mesh_color_network):

        self.material = threestudio.find(self.cfg.system.color_material_type)(self.cfg.system.color_material).to(self.device)
        self.background = threestudio.find(self.cfg.system.color_background_type)(self.cfg.system.color_background).to(self.device)
        self.renderer = threestudio.find(self.cfg.system.color_renderer_type)(
            self.cfg.system.color_renderer,
            geometry=mesh_color_network,              # mesh_color_network to get features
            material=self.material,
            background=self.background,
        ).to(self.device)
        self.exporter = threestudio.find(self.cfg.system.color_exporter_type)(
            self.cfg.system.color_exporter,
            geometry=mesh_color_network,              # mesh_color_network to get features
            material=self.material,
            background=self.background,
        )

    def collect_inputs(self, out, collect_inputs):
        inputs = [out[key] for key in collect_inputs]
        return torch.cat(inputs, dim=-1)
    
    def get_linear_value(self, values):

        start_step, start_value, end_value, end_step = values
        current_step = self.true_global_step
        value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        
        return value
    
    def get_camera_views(self, update_camera, azimuth_range=None, elevation_range=None):

        batch = self.camera_module.get_camera(azimuth_range, elevation_range)       # random views
        # data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        if update_camera:
            self.camera_module.update_step(self.true_global_step)
            if azimuth_range is not None and elevation_range is not None:
                print(f'*********** update camera views at step {self.true_global_step}, azimuth_range {azimuth_range}, elevation_range {elevation_range} ***********')
            else:
                print(f'*********** update camera views at step {self.true_global_step} ***********')

        return batch
    
    def get_anchor_camera_views(self, anchor_bin=10, elevation_range = [5, 30], infer_color=False):

        if infer_color:
            batch = self.camera_module.get_infer_anchor_camera(anchor_bin, elevation_range)       # fix elevation
        else:
            batch = self.camera_module.get_anchor_camera(anchor_bin, elevation_range)       # fixed views
        # data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        return batch
    
    def get_scene_anchor_camera_views(self, azimuth_bin=30, elevation_bin=30, camera_center=torch.zeros(3), big_fovy_range=[70, 85]):

        batch = self.camera_module.get_scene_anchor_camera(azimuth_bin, elevation_bin, camera_center, big_fovy_range)       # fixed views
        # data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        return batch

    def get_scene_camera_views(self, camera_center):

        batch = self.camera_module.get_scene_camera(camera_center)       # fixed views
        # data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        return batch
    
    def get_sds_loss(self, batch, sim_obj_idx):

        if not self.begin_color_prior:
            # NOTE: update occupancy grid before rendering
            self.renderer.update_step(epoch=None, global_step=self.true_global_step)

            # render images
            out = self.renderer(**batch, render_rgb=True)

        else:
            mesh = self.color_mesh_dict[sim_obj_idx]
            out = self.renderer(mesh, **batch)

        self._save_dir = os.path.join(self._root_save_dir, f'obj_{sim_obj_idx}')
        os.makedirs(self._save_dir, exist_ok=True)

        plot_interval = 100
        if self.true_global_step % plot_interval == 0:

            for idx in range(out['comp_rgb'].shape[0]):
                
                self.save_image_grid(
                    f"it{self.true_global_step:06d}_{idx}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_rgb"][idx],
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                        if "comp_rgb" in out
                        else []
                    )
                    + (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_normal_cam_white_vis"][idx],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                        if "comp_normal_cam_white_vis" in out
                        else []
                    )
                    + (
                        [
                            {
                                "type": "grayscale",
                                "img": out["vis_map"][idx],
                                "kwargs": {"cmap": "jet"},
                            }
                        ]
                        if "vis_map" in out
                        else []
                    )
                    + (
                        [
                            {
                                "type": "grayscale",
                                "img": (out["vis_map"][idx] > 0.5).float(),
                                "kwargs": {"cmap": "jet"},
                            }
                        ]
                        if "vis_map" in out
                        else []
                    ),
                    name="validation_step",
                    step=self.true_global_step,
                )

        if self.use_sd_prior:
            # Stable Diffusion
            collect_inps = self.cfg.system.guidance.collect_inputs
            guidance_inp = self.collect_inputs(out, collect_inps)
            self.guidance.update_step(None, self.true_global_step)                  # update timestep noise

            timestep = None

            if not self.begin_color_prior:
                guidance_inp = guidance_inp * 2.0 - 1.0

                guidance_out = self.guidance(
                    guidance_inp,
                    self.prompt_utils,
                    **batch,
                    rgb_as_latents=True,
                    timestep=timestep,
                    vis_mask = out['vis_map'] if 'vis_map' in out else None
                )
            else:
                guidance_out = self.guidance(
                    guidance_inp,
                    self.prompt_utils,
                    **batch,
                    rgb_as_latents=False,
                    timestep=timestep,
                    begin_color_prior=self.begin_color_prior,
                    color_vis_thresh=self.color_vis_thresh,
                    vis_mask = out['vis_map'] if 'vis_map' in out else None
                )

            loss_rgb_sd = guidance_out['loss_sds'] * self.cfg.system.loss.lambda_rgb_sds

        else:
            loss_rgb_sd = torch.tensor(0.0).to(self.device)

        sds_loss_dict = {
            'loss_rgb_sd': loss_rgb_sd,
        }

        return sds_loss_dict

    def export_color_mesh(self, obj_idx, save_root_path, save_name, prior_bbox_path=None):

        self._save_dir = save_root_path
        os.makedirs(self._save_dir, exist_ok=True)

        mesh = self.color_mesh_dict[obj_idx]
        exporter_output = self.exporter(mesh)
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)

            # recover color mesh original coordinates for each objects
            if prior_bbox_path is not None:
                out.params["mesh"] = self.recover_colormesh_coords(out.params["mesh"], prior_bbox_path)

            save_func(f"it{self.true_global_step}-export/{save_name}.obj", **out.params)

        mesh_save_path = os.path.join(self._save_dir, f'it{self.true_global_step}-export/{save_name}.obj')
        print(f'*********** export color mesh {obj_idx} to {mesh_save_path} at step {self.true_global_step} ***********')

    def recover_colormesh_coords(self, mesh, prior_bbox_path):

        with open(prior_bbox_path, 'r') as f:
            obj_bbox = json.load(f)

        x_min = obj_bbox[0][0] + 0.1
        y_min = obj_bbox[0][1] + 0.1
        z_min = obj_bbox[0][2] + 0.1
        x_max = obj_bbox[1][0] - 0.1
        y_max = obj_bbox[1][1] - 0.1
        z_max = obj_bbox[1][2] - 0.1
        x_len = x_max - x_min
        y_len = y_max - y_min
        z_len = z_max - z_min
        centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        centroid = torch.tensor(centroid).float().to(self.device)

        max_len = max(x_len, y_len, z_len)
        shape_scale = self.cfg.system.geometry.shape_init_params / max_len

        mesh.v_pos = mesh.v_pos / 2.0
        mesh.v_pos = mesh.v_pos / shape_scale
        mesh.v_pos = mesh.v_pos + centroid

        return mesh

