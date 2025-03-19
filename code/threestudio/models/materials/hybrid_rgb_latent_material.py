import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("hybrid-rgb-latent-material")
class HybridRGBLatentMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        requires_normal: bool = True
        requires_tangent: bool = False

    cfg: Config

    def configure(self) -> None:
        self.requires_normal = self.cfg.requires_normal
        self.requires_tangent = self.cfg.requires_tangent

    def forward(
        self, features: Float[Tensor, "B ... Nf"], **kwargs
    ) -> Float[Tensor, "B ... Nc"]:
        assert (
            features.shape[-1] == self.cfg.n_output_dims
        ), f"Expected {self.cfg.n_output_dims} output dims, only got {features.shape[-1]} dims input."
        color = features[..., :3]
        color = get_activation(self.cfg.color_activation)(color)
        return color

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:

        color = features[..., :3]
        color = get_activation(self.cfg.color_activation)(color)

        return {"albedo": color}
