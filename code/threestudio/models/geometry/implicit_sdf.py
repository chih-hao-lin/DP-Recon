import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

import threestudio
from threestudio.models.geometry.base import (BaseGeometry,
                                              BaseImplicitGeometry,
                                              contract_to_unisphere,)
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *


class PseudoSDF(nn.Module):
    def __init__(self, recon_sdf, shape_init_params):
        super(PseudoSDF, self).__init__()

        self.pseudo_sdf_network = recon_sdf
        self.obj_idx_list = None
        self.obj_bbox = None
        self.shape_scale = None
        self.shape_init_params = shape_init_params
        self.visgrid = None
        self.use_visgrid = False

    def convert_coord(self, x):
        '''
        obj_bbox: x_min - 0.1, y_min - 0.1, z_min - 0.1, x_max + 0.1, y_max + 0.1, z_max + 0.1
        NOTE: need to use obj bbox
        prior coord: [0, 1]
        object should align to [0.25, 0.75], some padding for background in camera view

        align stage:
        1.object --> [-0.25, 0.25]
        2.object + 0.5 --> [0.25, 0.75]

        input x is in [0, 1]
        '''
        x_min = self.obj_bbox[0][0] + 0.1
        y_min = self.obj_bbox[0][1] + 0.1
        z_min = self.obj_bbox[0][2] + 0.1
        x_max = self.obj_bbox[1][0] - 0.1
        y_max = self.obj_bbox[1][1] - 0.1
        z_max = self.obj_bbox[1][2] - 0.1
        x_len = x_max - x_min
        y_len = y_max - y_min
        z_len = z_max - z_min
        centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        centroid = torch.tensor(centroid).to(x.device)

        # shape_init_params = 0.5             # max_len*shape_scale = (1-0)*shape_init_params = 0.5
        max_len = max(x_len, y_len, z_len)
        self.shape_scale = self.shape_init_params / max_len       # sdf should apply shape_scale

        x = x - 0.5     # in [-0.5, 0.5]
        x = x / self.shape_scale    # in [-max_len, max_len]
        x = x + centroid

        return x
    
    def forward(self, x):
        if 0 not in self.obj_idx_list:                    # bg not need to scale
            x = self.convert_coord(x)
            obj_sdfs = self.pseudo_sdf_network(x)[:, self.obj_idx_list]
            output = torch.min(obj_sdfs, dim=1).values * self.shape_scale
        else:
            obj_sdfs = self.pseudo_sdf_network(x)[:, self.obj_idx_list]
            output = torch.min(obj_sdfs, dim=1).values
        
        return output
    
    def get_vis(self, x):
        if 0 not in self.obj_idx_list:                    # bg not need to scale
            x = self.convert_coord(x)

        with torch.no_grad():                               # no need to compute gradients
            max_trans = self.visgrid(x)                     # [num_points, 1]

        return max_trans.detach()

    def get_outputs(self, x):
        if 0 not in self.obj_idx_list:                    # bg not need to scale
            x = self.convert_coord(x)
            _, feature_vectors, gradients, _, sdf_raw = self.pseudo_sdf_network.get_outputs(x)
            obj_sdfs = sdf_raw[:, self.obj_idx_list]
            obj_sdf = torch.min(obj_sdfs, dim=1).values * self.shape_scale
        else:
            _, feature_vectors, gradients, _, sdf_raw = self.pseudo_sdf_network.get_outputs(x)
            obj_sdfs = sdf_raw[:, self.obj_idx_list]
            obj_sdf = torch.min(obj_sdfs, dim=1).values
        
        return obj_sdf, feature_vectors, gradients

class PseudoColor(nn.Module):
    def __init__(self, recon_color, shape_init_params):
        super(PseudoColor, self).__init__()

        self.pseudo_color_network = recon_color
        self.obj_idx_list = None
        self.obj_bbox = None
        self.shape_scale = None
        self.shape_init_params = shape_init_params

    def convert_coord(self, x):
        '''
        obj_bbox: x_min - 0.1, y_min - 0.1, z_min - 0.1, x_max + 0.1, y_max + 0.1, z_max + 0.1
        NOTE: need to use obj bbox
        prior coord: [0, 1]
        object should align to [0.25, 0.75], some padding for background in camera view

        align stage:
        1.object --> [-0.25, 0.25]
        2.object + 0.5 --> [0.25, 0.75]

        input x is in [0, 1]
        '''
        x_min = self.obj_bbox[0][0] + 0.1
        y_min = self.obj_bbox[0][1] + 0.1
        z_min = self.obj_bbox[0][2] + 0.1
        x_max = self.obj_bbox[1][0] - 0.1
        y_max = self.obj_bbox[1][1] - 0.1
        z_max = self.obj_bbox[1][2] - 0.1
        x_len = x_max - x_min
        y_len = y_max - y_min
        z_len = z_max - z_min
        centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        centroid = torch.tensor(centroid).to(x.device)

        # shape_init_params = 0.5             # max_len*shape_scale = (1-0)*shape_init_params = 0.5
        max_len = max(x_len, y_len, z_len)
        self.shape_scale = self.shape_init_params / max_len       # sdf should apply shape_scale

        x = x - 0.5     # in [-0.5, 0.5]
        x = x / self.shape_scale    # in [-max_len, max_len]
        x = x + centroid

        return x
    
    def forward(self, x, normals, ray_origins, feature_vectors):

        if 0 not in self.obj_idx_list:                    # bg not need to scale
            x = self.convert_coord(x)
            ray_origins = self.convert_coord(ray_origins)

        # calculate view_dirs
        view_dirs = (x - ray_origins).detach()
        view_dirs = F.normalize(view_dirs, dim=1)

        # pseudo_indices = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)
        rgb_flat = self.pseudo_color_network(x, normals, view_dirs, feature_vectors, indices=None)

        return rgb_flat


@threestudio.register("implicit-sdf")
class ImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

    cfg: Config

    def set_sdf_network(self, recon_sdf, shape_init_params):
        self.sdf_network = PseudoSDF(recon_sdf, shape_init_params)

    def set_obj_idx(self, obj_idx_list):
        self.sdf_network.obj_idx_list = obj_idx_list
        self.color_network.obj_idx_list = obj_idx_list
        self.obj_idx_list = obj_idx_list

    def set_obj_bbox(self, obj_bbox):
        self.sdf_network.obj_bbox = obj_bbox

    def set_color_network(self, recon_color, shape_init_params):
        self.color_network = PseudoColor(recon_color, shape_init_params)

    def configure(self) -> None:
        super().configure()
        # self.encoding = get_encoding(
        #     self.cfg.n_input_dims, self.cfg.pos_encoding_config
        # )
        # self.sdf_network = get_mlp(
        #     self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        # )

        # if self.cfg.n_feature_dims > 0:
        #     self.feature_network = get_mlp(
        #         self.encoding.n_output_dims,
        #         self.cfg.n_feature_dims,
        #         self.cfg.mlp_network_config,
        #     )

        # if self.cfg.normal_type == "pred":
        #     self.normal_network = get_mlp(
        #         self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
        #     )
        # if self.cfg.isosurface_deformable_grid:
        #     assert (
        #         self.cfg.isosurface_method == "mt"
        #     ), "isosurface_deformable_grid only works with mt"
        #     self.deformation_network = get_mlp(
        #         self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
        #     )

        self.finite_difference_normal_eps: Optional[float] = None

    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        if self.cfg.sdf_bias != 0.0:
            threestudio.warn(
                "shape_init and sdf_bias are both specified, which may lead to unexpected results."
            )

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            from pysdf import SDF

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm

        for _ in tqdm(
            range(1000),
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            points_rand = (
                torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
            )
            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def forward(
        self, points: Float[Tensor, "*N Di"], ray_origins: Float[Tensor, "*N Di"], output_normal: bool = False, output_vis: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        if 0 not in self.obj_idx_list:                    # bg not need to scale
            points = contract_to_unisphere(
                points, self.bbox, self.unbounded
            )  # points normalized to (0, 1)

        # enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        # sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf, feature_vectors, gradients = self.sdf_network.get_outputs(points)
        sdf = sdf.view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {"sdf": sdf}

        if output_vis:
            vis_trans = self.sdf_network.get_vis(points)
            output.update({"vis_trans": vis_trans})

        if self.cfg.n_feature_dims > 0:
            # features = self.feature_network(enc).view(
            #     *points.shape[:-1], self.cfg.n_feature_dims
            # )
            # features = torch.zeros(*points.shape[:-1], 3).to(points.device)
            features = self.color_network(points, gradients, ray_origins, feature_vectors)
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (
                        0.5
                        * (sdf_offset[..., 0::2, 0] - sdf_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
                sdf_grad = normal
            elif self.cfg.normal_type == "analytic":
                sdf_grad = -torch.autograd.grad(
                    sdf,
                    points_unscaled,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                )[0]
                normal = F.normalize(sdf_grad, dim=-1)
                if not grad_enabled:
                    sdf_grad = sdf_grad.detach()
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update(
                {"normal": normal, "shading_normal": normal, "sdf_grad": sdf_grad}
            )
        return output

    def forward_sdf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        if 0 not in self.obj_idx_list:                    # bg not need to scale
            points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        # sdf = self.sdf_network(
        #     self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        # ).reshape(*points.shape[:-1], 1)
        sdf = self.sdf_network(
            points.reshape(-1, self.cfg.n_input_dims)
        ).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf

    # def forward_field(
    #     self, points: Float[Tensor, "*N Di"]
    # ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
    #     points_unscaled = points
    #     points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
    #     # enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
    #     # sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)
    #     sdf = self.sdf_network(points.reshape(-1, self.cfg.n_input_dims)).reshape(*points.shape[:-1], 1)
    #     sdf = self.get_shifted_sdf(points_unscaled, sdf)
    #     deformation: Optional[Float[Tensor, "*N 3"]] = None
    #     # if self.cfg.isosurface_deformable_grid:
    #     #     deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)
    #     return sdf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        if 0 not in self.obj_idx_list:                    # bg not need to scale
            points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        # enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        # features = self.feature_network(enc).view(
        #     *points.shape[:-1], self.cfg.n_feature_dims
        # )
        features = torch.zeros(*points.shape[:-1], 3).to(points.device)
        out.update(
            {
                "features": features,
            }
        )
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (
                    current_level - 1
                )
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    threestudio.info(
                        f"Update finite_difference_normal_eps to {grid_size}"
                    )
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(
                    f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}"
                )

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry, cfg: Optional[Union[dict, DictConfig]] = None, **kwargs
    ) -> BaseGeometry:
        return other
