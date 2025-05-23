import math
import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.ops import chunk_batch, validate_empty_rays
from threestudio.utils.typing import *


class PseudoVolsdfDensity(nn.Module):
    def __init__(self, recon_density):
        super(PseudoVolsdfDensity, self).__init__()
        
        self.pseudo_density = recon_density

    def forward(self, sdf):
        
        return self.pseudo_density(sdf)


@threestudio.register("neus-volume-renderer")
class NeuSVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        learned_variance_init: float = 0.3
        cos_anneal_end_steps: int = 0
        use_volsdf: bool = False
        occ_grid_res: int = 32  # 32
        depth_norm_radius: float = 1.0

        near_plane: float = 0.0
        far_plane: float = 1e10

        # in ['occgrid', 'importance']
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for importance
        num_samples_per_ray_importance: int = 64

        return_comp_normal: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
        recon_density,
    ) -> None:
        super().configure(geometry, material, background)
        # self.variance = LearnedVariance(self.cfg.learned_variance_init)
        # if self.cfg.estimator == "occgrid":
        #     self.estimator = nerfacc.OccGridEstimator(
        #         roi_aabb=self.bbox.view(-1), resolution=self.cfg.occ_grid_res, levels=1
        #     )
        #     if not self.cfg.grid_prune:
        #         self.estimator.occs.fill_(True)
        #         self.estimator.binaries.fill_(True)
        #     self.render_step_size = (
        #         1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        #     )
        #     self.randomized = self.cfg.randomized
        # elif self.cfg.estimator == "importance":
        #     self.estimator = ImportanceEstimator()
        # else:
        #     raise NotImplementedError(
        #         "unknown estimator, should be in ['occgrid', 'importance']"
        #     )
        self.cos_anneal_ratio = 1.0

        self.density = PseudoVolsdfDensity(recon_density)

        print(f'use {self.cfg.estimator} estimator for ray sampling.')

    def create_estimator(self):
        if self.cfg.estimator == "occgrid":
            estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=self.cfg.occ_grid_res, levels=1
            )
            if not self.cfg.grid_prune:
                estimator.occs.fill_(True)
                estimator.binaries.fill_(True)
            self.render_step_size = (
                1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
            )
            self.randomized = self.cfg.randomized
        elif self.cfg.estimator == "importance":
            estimator = ImportanceEstimator()
            self.randomized = self.cfg.randomized
        else:
            raise NotImplementedError(
                "unknown estimator, should be in ['occgrid', 'importance']"
            )
        
        return estimator
    
    def init_estimator(self, obj_estimator):
        self.estimator = obj_estimator

    def get_alpha(self, sdf, normal, dirs, dists):
        # inv_std = self.variance(sdf)
        if self.cfg.use_volsdf:
            # alpha = torch.abs(dists.detach()) * volsdf_density(sdf, inv_std)
            alpha = torch.abs(dists.detach()) * self.density(sdf)
            alpha = 1 - torch.exp(-alpha)
        else:
            true_cos = (dirs * normal).sum(-1, keepdim=True)
            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(
                F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)
                + F.relu(-true_cos) * self.cos_anneal_ratio
            )  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha

    def depth_to_disparity(
        self, depth, valid_mask, target_min=50, target_max=255, pad_value=10
    ):
        target_min = target_min / 255.0
        target_max = target_max / 255.0
        pad_value = pad_value / 255.0

        depth_tmp = depth

        unvalid_mask = ~valid_mask

        depth_max = depth_tmp[
            valid_mask
        ].max()  # self.cfg.camera_distance_range[1] + self.cfg.radius
        depth_min = depth_tmp[valid_mask].min()  # 0.1
        depth_tmp = depth_max - depth_tmp  # reverse
        depth_tmp /= depth_max - depth_min  # [0, 1]
        depth_tmp = depth_tmp.clamp(0, 1)

        depth_tmp = depth_tmp * (target_max - target_min) + target_min
        depth_tmp[unvalid_mask] = pad_value

        return depth_tmp

    def convert_pose(self, C2W):
        flip_yz = torch.eye(4, device=C2W.device, dtype=C2W.dtype)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        bs = C2W.shape[0]
        flip_yz = flip_yz.unsqueeze(0).repeat(bs, 1, 1)
        C2W = torch.matmul(C2W, flip_yz)
        return C2W

    def convert_normal_to_cam_space(self, normal: torch.Tensor, c2w):
        """
        normal:[B, N, 3, 1]
        c2w: [B, 4, 4]
        reuturn : [BN, 3]
        """
        batch_size = c2w.shape[0]

        # c2w = self.convert_pose(c2w).float()

        assert normal.ndim == 2
        w2c = torch.inverse(c2w)

        # normal = normal.reshape(batch_size, -1, 3, 1)  # [b, n, 3, 1]
        # normal = torch.matmul(w2c[:, :3, :3].unsqueeze(1), normal).reshape(-1, 3)  # [b, 1, 3, 3], [b, n, 3, 1]

        normal = normal.reshape(batch_size, -1, 3)  # [b, n, 3]
        rotate: Float[Tensor, "B 4 4"] = w2c[..., :3, :3]
        camera_normal = normal @ rotate.permute(0, 2, 1)
        # pixel space flip axis so we need built negative y-axis normal
        flip_x = torch.eye(3).to(w2c)
        flip_x[0, 0] = -1

        camera_normal = camera_normal @ flip_x[None, ...]
        camera_normal = camera_normal.reshape(-1, 3)

        return camera_normal

    def depth_normalization(
        self,
        depth,
        valid_mask,
        depth_min=None,
        depth_max=None,
        target_min=50,
        target_max=255,
        pad_value=10,
        camera_distances=None,
        radius=1,
    ):
        depth_tmp = depth
        camera_distances = camera_distances.reshape(-1, 1, 1, 1)
        unvalid_mask = ~valid_mask

        near = camera_distances - radius * math.sqrt(3)
        far = camera_distances + radius * math.sqrt(3)

        depth_norm = (far - depth_tmp) / (far - near)

        depth_norm = torch.clamp(depth_norm, 0, 1)

        return depth_norm

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]

        if self.cfg.estimator == "occgrid":

            def alpha_fn(t_starts, t_ends, ray_indices):
                t_starts, t_ends = t_starts[..., None], t_ends[..., None]
                t_origins = rays_o_flatten[ray_indices]
                t_positions = (t_starts + t_ends) / 2.0
                t_dirs = rays_d_flatten[ray_indices]
                positions = t_origins + t_dirs * t_positions
                if self.training:
                    sdf = self.geometry.forward_sdf(positions)[..., 0]
                else:
                    sdf = chunk_batch(
                        self.geometry.forward_sdf,
                        self.cfg.eval_chunk_size,
                        positions,
                    )[..., 0]

                # inv_std = self.variance(sdf)
                if self.cfg.use_volsdf:
                    # alpha = self.render_step_size * volsdf_density(sdf, inv_std)
                    alpha = self.render_step_size * self.density(sdf)
                    alpha = 1 - torch.exp(-alpha)
                else:
                    estimated_next_sdf = sdf - self.render_step_size * 0.5
                    estimated_prev_sdf = sdf + self.render_step_size * 0.5
                    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                    next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                    p = prev_cdf - next_cdf
                    c = prev_cdf
                    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

                return alpha

            if not self.cfg.grid_prune:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        alpha_fn=None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0,
                    )
            else:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        alpha_fn=alpha_fn if self.cfg.prune_alpha_threshold else None,
                        near_plane=self.cfg.near_plane,
                        far_plane=self.cfg.far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                    )
        elif self.cfg.estimator == "importance":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                if self.cfg.use_volsdf:
                    t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                    t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                    positions: Float[Tensor, "Nr Ns 3"] = (
                        t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                    )
                    with torch.no_grad():
                        geo_out = chunk_batch(
                            proposal_network,
                            self.cfg.eval_chunk_size,
                            positions.reshape(-1, 3),
                        )
                        # inv_std = self.variance(geo_out)
                        # density = volsdf_density(geo_out, inv_std)
                        density = self.density(geo_out)
                    return density.reshape(positions.shape[:2])
                else:
                    raise ValueError(
                        "Currently only VolSDF supports importance sampling."
                    )

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.geometry.forward_sdf)],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError

        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if self.training:
            output_vis = self.geometry.sdf_network.use_visgrid
            geo_out = self.geometry(positions, ray_origins=t_origins, output_normal=True, output_vis=output_vis)
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=True,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d
            )

        # grad or normal?
        alpha: Float[Tensor, "Nr 1"] = self.get_alpha(
            geo_out["sdf"], geo_out["normal"], t_dirs, t_intervals
        )

        weights: Float[Tensor, "Nr 1"]
        weights_, _ = nerfacc.render_weight_from_alpha(
            alpha[..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )
        vis_map: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=geo_out['vis_trans'], ray_indices=ray_indices, n_rays=n_rays
        )
        vis_map[vis_map > 1.0] = 1.0                # [0, 1]
        vis_mask = (vis_map > 0.5).reshape(batch_size, -1)
        obj_mask = (opacity > 0.00001).reshape(batch_size, -1)
        vis_rate = torch.sum(vis_mask, dim=-1) / torch.sum(obj_mask, dim=-1)

        if bg_color is None:
            bg_color = comp_rgb_bg

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "opacity_rerange": (opacity.view(batch_size, height, width, 1) + 1.0) / 2,
            "depth": depth.view(batch_size, height, width, 1),
            "vis_map": vis_map.view(batch_size, height, width, 1),
            "vis_rate": vis_rate,
        }

        radius = self.cfg.depth_norm_radius
        # radius = 0.75
        depth_bg_max = kwargs["camera_distances"].reshape(
            -1, 1, 1, 1
        ) + radius * math.sqrt(3)
        # depth_bg_max = (kwargs['camera_distances'] + self.cfg.radius)
        # depth_bg_max = depth_bg_max.reshape(-1, 1, 1, 1)

        depth = out["depth"] * out["opacity"] + (1 - out["opacity"]) * depth_bg_max
        # depth = out["depth"]

        # print(
        #     kwargs["camera_distances"].detach().cpu().numpy().min(),
        #     kwargs["camera_distances"].detach().cpu().numpy().max(),
        #     out["depth"].detach().cpu().numpy().min(),
        #     out["depth"].detach().cpu().numpy().max(),
        #     depth.detach().cpu().numpy().min(),
        #     depth.detach().cpu().numpy().max(),
        # )
        disparity = self.depth_normalization(
            depth,
            valid_mask=(out["depth"] > 0),
            target_min=20,
            pad_value=0,
            camera_distances=kwargs["camera_distances"],
            radius=radius,
        )

        out["disparity"] = disparity

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )

            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal_mask = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization

                    bg_normal = 0.5 * torch.ones_like(comp_normal)
                    bg_normal[:, 2] = 1.0
                    comp_normal_vis = (comp_normal + 1.0) / 2.0 * opacity + (
                        1 - opacity
                    ) * bg_normal
                    # convert to cam space
                    comp_normal_cam = self.convert_normal_to_cam_space(
                        comp_normal, kwargs["c2w"]
                    )
                    bg_normal = 0.5 * torch.ones_like(comp_normal_cam)
                    bg_normal[:, 2] = 1.0
                    comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity + (
                        1 - opacity
                    ) * bg_normal

                    bg_normal_white = torch.ones_like(comp_normal_cam)
                    comp_normal_cam_white_vis = (
                        comp_normal_cam + 1.0
                    ) / 2.0 * opacity + (1 - opacity) * bg_normal_white

                    out.update(
                        {
                            "comp_normal": comp_normal_mask.view(
                                batch_size, height, width, 3
                            ),
                            "comp_normal_vis": comp_normal_vis.view(
                                batch_size, height, width, 3
                            ),
                            "comp_normal_cam_vis": comp_normal_cam_vis.view(
                                batch_size, height, width, 3
                            ),
                            "comp_normal_cam_white_vis": comp_normal_cam_white_vis.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )

        else:
            if "normal" in geo_out:
                comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal_mask = (
                    (comp_normal + 1.0) / 2.0 * opacity
                )  # for visualization

                bg_normal = 0.5 * torch.ones_like(comp_normal)
                bg_normal[:, 2] = 1.0
                comp_normal_vis = (comp_normal + 1.0) / 2.0 * opacity + (
                    1 - opacity
                ) * bg_normal
                # convert to cam space
                comp_normal_cam = self.convert_normal_to_cam_space(
                    comp_normal, kwargs["c2w"]
                )
                bg_normal = 0.5 * torch.ones_like(comp_normal_cam)
                bg_normal[:, 2] = 1.0
                comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity + (
                    1 - opacity
                ) * bg_normal

                bg_normal_white = torch.ones_like(comp_normal_cam)
                comp_normal_cam_white_vis = (
                    comp_normal_cam + 1.0
                ) / 2.0 * opacity + (1 - opacity) * bg_normal_white

                out.update(
                    {
                        "comp_normal": comp_normal_mask.view(
                            batch_size, height, width, 3
                        ),
                        "comp_normal_vis": comp_normal_vis.view(
                            batch_size, height, width, 3
                        ),
                        "comp_normal_cam_vis": comp_normal_cam_vis.view(
                            batch_size, height, width, 3
                        ),
                        "comp_normal_cam_white_vis": comp_normal_cam_white_vis.view(
                            batch_size, height, width, 3
                        ),
                    }
                )
        # out.update({"inv_std": self.variance.inv_std})
        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.cos_anneal_ratio = (
            1.0
            if self.cfg.cos_anneal_end_steps == 0
            else min(1.0, global_step / self.cfg.cos_anneal_end_steps)
        )
        if self.cfg.estimator == "occgrid":
            if self.cfg.grid_prune:

                def occ_eval_fn(x):
                    sdf = self.geometry.forward_sdf(x)
                    # inv_std = self.variance(sdf)
                    if self.cfg.use_volsdf:
                        # alpha = self.render_step_size * volsdf_density(sdf, inv_std)
                        alpha = self.render_step_size * self.density(sdf)
                        alpha = 1 - torch.exp(-alpha)
                    else:
                        estimated_next_sdf = sdf - self.render_step_size * 0.5
                        estimated_prev_sdf = sdf + self.render_step_size * 0.5
                        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                        next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                        p = prev_cdf - next_cdf
                        c = prev_cdf
                        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                    return alpha

                if self.training and not on_load_weights:
                    self.estimator.update_every_n_steps(
                        step=global_step, occ_eval_fn=occ_eval_fn
                    )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()
