import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch
import cv2
import json

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import matplotlib.pyplot as plt
import numpy as np
from utils.pano_camera_tools import pano_2_pers

from torch import vmap
from prior import PriorModule

from omegaconf import OmegaConf
from camera_views import CameraViews
from threestudio.models.networks import get_encoding
import tinycudann as tcnn

from threestudio.utils.ops import get_mvp_matrix, get_projection_matrix

# NOTE: get_projection_matrix from threestudio.utils.ops need cx = w/2, cy = h/2, not suitable for more general cases
def get_projection_matrix_my(K, w, h, znear=0.1, zfar=1000.0):

    fx_opengl = K[0, 0, 0] / w
    fy_opengl = K[0, 1, 1] / h
    cx_opengl = K[0, 0, 2] / w
    cy_opengl = K[0, 1, 2] / h

    proj_mtx = torch.tensor([
        [2 * fx_opengl, 0, 1 - 2 * cx_opengl, 0],
        [0, 2 * fy_opengl, 1 - 2 * cy_opengl, 0],
        [0, 0, -(zfar + znear) / (zfar - znear), -2 * zfar * znear / (zfar - znear)],
        [0, 0, -1, 0]
    ], dtype=torch.float32)

    # flip y axis
    flip_y = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    proj_mtx = flip_y @ proj_mtx

    proj_mtx = proj_mtx.unsqueeze(0)
    
    return proj_mtx

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=True,
            sigmoid = 10
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        print(multires, dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out
        self.sigmoid = sigmoid

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    torch.nn.init.normal_(lin.weight[:1, :], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[:1], bias) 
                    # inner objects with SDF initial with negative value inside and positive value outside, ~0.6 radius of background
                    torch.nn.init.normal_(lin.weight[1:,:], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[1:], -0.6*bias) 

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.pool = nn.MaxPool1d(self.d_out, return_indices=True)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        g = []
        for idx in range(y.shape[1]):
            gradients = torch.autograd.grad(
                outputs=y[:, idx:idx+1],
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            g.append(gradients)
        g = torch.cat(g)
        # add the gradient of minimum sdf
        # sdf = -self.pool(-y.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-y.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        g_min_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        g = torch.cat([g, g_min_sdf])
        return g

    def get_outputs(self, x, beta=None):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))
        if beta == None:
            semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        else:
            semantic = 0.5/beta *torch.exp(-sdf_raw.abs()/beta)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        # sdf = -self.pool(-sdf) # get the minium value of sdf  if bound apply in the final 
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf.expand(sdf.shape))
        # sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1) # get the minium value of sdf if bound apply before min
        sdf, indices = self.pool(-sdf.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        return sdf
    
    def get_sdf_raw(self, x):
        return self.forward(x)[:, :self.d_out]
    
    def get_object_sdf_vals(self, x, idx):
        sdf = self.forward(x)[:, idx]
        return sdf
    
    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw[:, idx]

    def get_shift_sdf_raw(self, x):
        sdf_raw = self.forward(x)[:, :self.d_out]
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        # shift raw sdf
        pos_min_sdf = -sdf          # other object sdf must bigger than -sdf
        pos_min_sdf_expand = pos_min_sdf.expand_as(sdf_raw)
        shift_mask = (sdf < 0)
        shift_mask_expand = shift_mask.expand_as(sdf_raw)

        shift_sdf_raw = torch.where(shift_mask_expand, torch.max(sdf_raw, pos_min_sdf_expand), sdf_raw)
        shift_sdf_raw[torch.arange(indices.size(0)), indices.squeeze()] = sdf.squeeze()

        return shift_sdf_raw

from hashencoder.hashgrid import HashEncoder
class ObjectImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0, # radius of the sphere in geometric initialization
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True, # use hash grid embedding or not, if not, it is a pure MLP with sin/cos embedding
            sigmoid = 20
    ):
        super().__init__()
        
        self.d_out = d_out
        self.sigmoid = sigmoid
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim
        
        print(f"[INFO]: using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"[INFO]: resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        # print("network architecture")
        # print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    torch.nn.init.normal_(lin.weight[:1, :], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[:1], bias)
                    # inner objects with SDF initial with negative value inside and positive value outside, ~0.5 radius of background
                    torch.nn.init.normal_(lin.weight[1:,:], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[1:], -0.5*bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

        self.pool = nn.MaxPool1d(self.d_out, return_indices=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            # assert torch.max(input / self.divide_factor)<1 and torch.min(input / self.divide_factor)>-1, 'range out of [-1, 1], max: {}, min: {}'.format(torch.max(input / self.divide_factor),  torch.min(input / self.divide_factor))
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))

        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        f = lambda v: torch.autograd.grad(outputs=y,
                    inputs=x,
                    grad_outputs=v.repeat(y.shape[0], 1),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        
        N = torch.eye(y.shape[1], requires_grad=False).to(y.device)
        
        # start_time = time.time()
        if self.use_grid_feature: # using hashing grid feature, cannot support vmap now
            g = torch.cat([torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=idx.repeat(y.shape[0], 1),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0] for idx in N.unbind()])
        # torch.cuda.synchronize()
        # print("time for computing gradient by for loop: ", time.time() - start_time, "s")
                
        # using vmap for batched gradient computation, if not using grid feature (pure MLP)
        else:
            g = vmap(f, in_dims=1)(N).reshape(-1, 3)
        
        # add the gradient of scene sdf
        # sdf = -self.pool(-y.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-y.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        g_min_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        g = torch.cat([g, g_min_sdf])
        return g

    def get_outputs(self, x, beta=None):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
        #     sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))

        if beta == None:
            semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        else:
            # change semantic to the gradianct of density
            semantic = 1/beta * (0.5 + 0.5 * sdf_raw.sign() * torch.expm1(-sdf_raw.abs() / beta))
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        feature_vectors = output[:, self.d_out:]

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_sdf_vals(self, x):
        sdf_raw = self.forward(x)[:,:self.d_out]
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        return sdf

    def get_sdf_raw(self, x):
        return self.forward(x)[:, :self.d_out]
    
    def get_object_sdf_vals(self, x, idx):
        sdf = self.forward(x)[:, idx]
        return sdf

    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw[:, idx]
    
    def get_shift_sdf_raw(self, x):
        sdf_raw = self.forward(x)[:, :self.d_out]
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        # shift raw sdf
        pos_min_sdf = -sdf          # other object sdf must bigger than -sdf
        pos_min_sdf_expand = pos_min_sdf.expand_as(sdf_raw)
        shift_mask = (sdf < 0)
        shift_mask_expand = shift_mask.expand_as(sdf_raw)

        shift_sdf_raw = torch.where(shift_mask_expand, torch.max(sdf_raw, pos_min_sdf_expand), sdf_raw)
        shift_sdf_raw[torch.arange(indices.size(0)), indices.squeeze()] = sdf.squeeze()

        return shift_sdf_raw

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self, verbose=False):
        if verbose:
            print("[INFO]: grid parameters", len(list(self.encoding.parameters())))
            for p in self.encoding.parameters():
                print(p.shape)
        return self.encoding.parameters()


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        # print("rendering network architecture:")
        # print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        x = self.sigmoid(x)
        return x

class MeshColorNetwork(nn.Module):
    def __init__(self, dim_in, dim_out, grid_config, shape_init_params, prior_bbox_root_path, n_neurons=64, n_hidden_layers=1):
        super().__init__()

        config_dict = OmegaConf.to_container(grid_config, resolve=True)
        self.pos_encoding = tcnn.Encoding(dim_in, config_dict, dtype=torch.float32)

        self.n_neurons, self.n_hidden_layers = n_neurons, n_hidden_layers
        layers = [
            self.make_linear(self.pos_encoding.n_output_dims, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)

        self.obj_idx_list = None
        self.obj_bbox = None
        self.shape_init_params = shape_init_params
        self.visgrid = None

        self.prior_bbox_root_path = prior_bbox_root_path

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
        NOTE: small different with NeRF implementation, inverse transformation with init_color_mesh
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

        max_len = max(x_len, y_len, z_len)
        shape_scale = self.shape_init_params / max_len       # sdf should apply shape_scale

        x = x / 2.0
        x = x / shape_scale
        x = x + centroid

        return x

    def forward(self, x, output_normal=False):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            if 0 not in self.obj_idx_list:                    # bg not need to scale
                x = self.convert_coord(x)
            x = self.pos_encoding(x)                        # NOTE: freeze the positional encoding, no need to update
            features = self.layers(x)
        return {"features": features}
    
    def get_vis(self, x):
        if 0 not in self.obj_idx_list:                    # bg not need to scale
            x = self.convert_coord(x)

        with torch.no_grad():                               # no need to compute gradients
            max_trans = self.visgrid(x)                     # [num_points, 1]

        return max_trans.detach()
    
    def set_obj_idx_list_bbox(self, obj_id):
        # NOTE: not use object groups

        self.obj_idx_list = [obj_id]

        if obj_id == 0:
            self.obj_bbox = None
        else:
            bbox_json_path = os.path.join(self.prior_bbox_root_path, f'bbox_{obj_id}.json')
            with open(bbox_json_path, 'r') as f:
                obj_bbox = json.load(f)

            self.obj_bbox = obj_bbox

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)


class DPReconNetwork(nn.Module):
    def __init__(self, conf, prior_yaml, plots_dir, ft_folder, obj_num, prompt_path):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.use_bg_reg = conf.get_bool('use_bg_reg', default=False)
        self.render_bg_iter = conf.get_int('render_bg_iter', default=10)
        self.begin_bg_surface_obj_reg_iter = conf.get_int('begin_bg_surface_obj_reg_iter', default=40000)
        self.use_virtual_camera_views = conf.get_bool('use_virtual_camera_views', default=True)         # for bg smoothing

        Grid_MLP = conf.get_bool('Grid_MLP', default=False)
        self.Grid_MLP = Grid_MLP
        if Grid_MLP: 
            self.implicit_network = ObjectImplicitNetworkGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))    
        else:
            self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        
        self.num_semantic = conf.get_int('implicit_network.d_out')

        self.plots_dir = plots_dir
        self.ft_folder = ft_folder
        self.prior_obj_idx_list = list(range(obj_num))
        self.prior_idx_iter = 0
        self.group_list = {key:[key] for key in self.prior_obj_idx_list}

        self.use_prior = conf.get_bool('prior.use_prior', default=False)
        self.prior_stage_milestones = conf.get_int('prior.prior_stage_milestones')
        print(f'******** use_prior: {self.use_prior}, prior_stage_milestones: {self.prior_stage_milestones} ********')
        self.begin_prior = False
        self.begin_color_prior = False
        if self.use_prior:
            self.prior = PriorModule(
                prior_yaml = prior_yaml,
                recon_sdf=self.implicit_network, 
                recon_color=self.rendering_network,
                recon_density=self.density,
                prior_obj_idx_list=self.prior_obj_idx_list,
                prompt_path=prompt_path,
                plots_dir=self.plots_dir,
            )
            self.color_vis_thresh = self.prior.color_vis_thresh
            self.bg_vis_thresh = self.prior.bg_vis_thresh
            self.mesh_color_network = MeshColorNetwork(
                dim_in=3, dim_out=3,
                grid_config=self.prior.cfg.system.geometry.pos_encoding_config,
                shape_init_params=self.prior.cfg.system.geometry.shape_init_params,
                prior_bbox_root_path=os.path.join(self.plots_dir, 'prior_bbox'),
            )
        else:
            # cfg_file = 'geo-iso-nerf.yaml'
            # cfg = OmegaConf.load(cfg_file)
            # self.scene_camera_module = CameraViews(cfg.data)
            # self.scene_camera_module.batch_size = 4
            raise ValueError('No prior module is used, please check the configuration file')

    def forward(self, input, indices, iter_step=-1, render_obj_idx=None, visgrid=None):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        cam_loc_temp = cam_loc.cpu().clone()[0]          # for novel camera views
        
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_outputs(points_flat, beta=None)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)

        if render_obj_idx is not None:
            obj_sdf = sdf_raw[:, render_obj_idx].unsqueeze(1)
            weights, transmittance, dists = self.volume_rendering(z_vals, obj_sdf) 
        else:
            weights, transmittance, dists = self.volume_rendering(z_vals, sdf) 

        # rendering the occlusion-awared object opacity
        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf_raw).sum(-1).transpose(0, 1)


        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1)*semantic, 1)
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values
        
        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb':rgb,
            'semantic_values': semantic_values, # here semantic value calculated as in ObjectSDF
            'object_opacity': object_opacity, 
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'points_flat': points_flat,
            'dirs_flat': dirs_flat,
            'transmittance': transmittance,
            'feature_vectors': feature_vectors,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels 
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
            
            grad_theta = self.implicit_network.gradient(eikonal_points)

            sample_sdf = self.implicit_network.get_sdf_raw(eikonal_points)
            sdf_value = self.implicit_network.get_sdf_vals(eikonal_points)
            output['sample_sdf'] = sample_sdf
            output['sample_minsdf'] = sdf_value
            
            # split gradient to eikonal points and heighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]

            # # use bg surface for regularization
            # if iter_step > self.begin_bg_surface_obj_reg_iter:                   # start to use bg surface for regularization
            #     surf_bg_z_vals = self.ray_sampler.ray_marching_surface(self, ray_dirs, cam_loc, idx=0) # [N, 1]
            #     # the sdf value of objects that behind bg surface
            #     bg_surf_back_mask = z_vals > surf_bg_z_vals # [1024, 98]
            #     sdf_all = sdf_raw.reshape(z_vals.shape[0], z_vals.shape[1], -1)
            #     objs_sdfs_bg_back = sdf_all[bg_surf_back_mask][..., 1:]  # [K, num_semantics-1]

            #     output['obj_sdfs_behind_bg'] = objs_sdfs_bg_back
        
        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        # only for render the background depth and normal
        iter_check = iter_step % self.render_bg_iter
        if self.use_bg_reg and iter_check == 0:

            if self.use_virtual_camera_views:

                # get novel camera pose
                if self.use_prior:
                    camera_batch = self.prior.get_scene_camera_views(camera_center=cam_loc_temp)
                else:
                    camera_batch = self.scene_camera_module.get_scene_camera(camera_center=cam_loc_temp)
                c2w = camera_batch['c2w'][0].cuda()                 # [4, 4], use the first camera view

                # NOTE: nerf use opengl coordinate system
                #       threestudio use colmap coordinate system (opencv coordinate system)

                # colmap to opengl
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1

                pose = c2w.unsqueeze(0).cuda()      # use novel camera pose, camera intrinsics remains the same

            # construct patch uv
            patch_size = 32
            n_patches = 1

            x0 = np.random.randint(0, 384 - patch_size + 1, size=(n_patches, 1, 1))         # NOTE: fix image resolution as 384
            y0 = np.random.randint(0, 384 - patch_size + 1, size=(n_patches, 1, 1))
            xy0 = np.concatenate([x0, y0], axis=-1)
            patch_idx = xy0 + np.stack(np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),axis=-1).reshape(1, -1, 2)
            uv0 = torch.from_numpy(patch_idx).float().reshape(1, -1, 2).float().cuda()
            ray_dirs0, cam_loc0 = rend_util.get_camera_params(uv0, pose, intrinsics)

            # we should use unnormalized ray direction for depth
            ray_dirs0_tmp, _ = rend_util.get_camera_params(uv0, torch.eye(4).to(pose.device)[None], intrinsics)
            depth_scale0 = ray_dirs0_tmp[0, :, 2:]
            
            batch_size, num_pixels, _ = ray_dirs0.shape

            cam_loc0 = cam_loc0.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
            ray_dirs0 = ray_dirs0.reshape(-1, 3)

            bg_z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs0, cam_loc0, self, idx=0)
            N_samples_bg = bg_z_vals.shape[1]

            bg_points = cam_loc0.unsqueeze(1) + bg_z_vals.unsqueeze(2) * ray_dirs0.unsqueeze(1)
            bg_points_flat = bg_points.reshape(-1, 3)
            scene_sdf, _, bg_gradients, scene_semantic, bg_sdf = self.implicit_network.get_specific_outputs(bg_points_flat, 0)
            
            bg_weight, _, _ = self.volume_rendering(bg_z_vals, bg_sdf)

            # NOTE: semantic should use scene sdf for volume rendering
            scene_weight, _, _ = self.volume_rendering(bg_z_vals, scene_sdf)
            scene_semantic = scene_semantic.reshape(-1, N_samples_bg, self.num_semantic)
            bg_semantic_value = torch.sum(scene_weight.unsqueeze(-1)*scene_semantic, 1)
            bg_mask = torch.argmax(bg_semantic_value, dim=-1, keepdim=True)
            output['bg_mask'] = bg_mask

            bg_depth_values = torch.sum(bg_weight * bg_z_vals, 1, keepdims=True) / (bg_weight.sum(dim=1, keepdims=True) +1e-8)
            bg_depth_values = depth_scale0 * bg_depth_values 
            output['bg_depth_values'] = bg_depth_values

            # compute bg normal map
            bg_normals = bg_gradients / (bg_gradients.norm(2, -1, keepdim=True) + 1e-6)
            bg_normals = bg_normals.reshape(-1, N_samples_bg, 3)
            bg_normal_map = torch.sum(bg_weight.unsqueeze(-1) * bg_normals, 1)
            bg_normal_map = rot @ bg_normal_map.permute(1, 0)
            bg_normal_map = bg_normal_map.permute(1, 0).contiguous()
            output['bg_normal_map'] = bg_normal_map

            # try:
            #     # use bg surface for regularization
            #     if iter_step > self.begin_bg_surface_obj_reg_iter:                   # start to use bg surface for regularization
            #         surf_bg_z_vals = self.ray_sampler.ray_marching_surface(self, ray_dirs0, cam_loc0, idx=0) # [N, 1]
            #         obj_z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs0, cam_loc0, self)
            #         obj_points = cam_loc0.unsqueeze(1) + obj_z_vals.unsqueeze(2) * ray_dirs0.unsqueeze(1)
            #         obj_points_flat = obj_points.reshape(-1, 3)
            #         _, _, _, _, obj_sdf_raw = self.implicit_network.get_outputs(obj_points_flat, beta=None)
            #         # the sdf value of objects that behind bg surface
            #         bg_surf_back_mask = obj_z_vals > surf_bg_z_vals # [1024, 98]
            #         obj_sdf_all = obj_sdf_raw.reshape(obj_z_vals.shape[0], obj_z_vals.shape[1], -1)
            #         objs_sdfs_bg_back = obj_sdf_all[bg_surf_back_mask][..., 1:]  # [K, num_semantics-1]

            #         output['obj_sdfs_behind_bg'] = objs_sdfs_bg_back

            # except:
            #     print('Error in using bg surface for regularization')

        if self.training and self.begin_prior:

            list_idx = self.prior_idx_iter % (len(self.prior_obj_idx_list))
            sim_obj_idx = self.prior_obj_idx_list[list_idx]
            group_prior_obj_list = self.group_list[sim_obj_idx]

            update_state = (sim_obj_idx == self.prior_obj_idx_list[-1])          # update after finish all object index

            self.prior_idx_iter += 1

            print(f'************** Prior object index {sim_obj_idx}: {group_prior_obj_list} **************')

            self.prior.set_prompt_renderer_idx(sim_obj_idx, group_prior_obj_list)

            # load object bbox(prior bbox only define once)
            if not os.path.exists(os.path.join(self.plots_dir, 'prior_bbox')):        # use object bbox
                ValueError('Please define object bbox first!')

            bbox_json_path = os.path.join(self.plots_dir, 'prior_bbox', f'bbox_{sim_obj_idx}.json')
            with open(bbox_json_path, 'r') as f:
                obj_bbox = json.load(f)

            self.prior.geometry.sdf_network.obj_bbox = obj_bbox
            self.prior.geometry.color_network.obj_bbox = obj_bbox
            if visgrid is not None:
                self.prior.geometry.sdf_network.use_visgrid = True
                self.prior.geometry.sdf_network.visgrid = visgrid
            
            if 0 in group_prior_obj_list:           # prior with bg
                camera_batch = self.prior.get_scene_camera_views(camera_center=cam_loc_temp)
            else:

                if iter_step < self.prior_stage_milestones:
                    azimuth_range = [-180, 180]
                    elevation_range = [5, 30]
                else:
                    azimuth_range = [-180, 180]
                    elevation_range = [5, 80]

                camera_batch = self.prior.get_camera_views(update_state, azimuth_range, elevation_range)

            sds_loss_dict = self.prior.get_sds_loss(camera_batch, sim_obj_idx)

            if update_state:                                       # update after finish all object index
                self.prior.true_global_step += 1
                self.prior.global_step += 1

            output['loss_rgb_sd'] = sds_loss_dict['loss_rgb_sd']
        
        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights, transmittance, dists

    def occlusion_opacity(self, z_vals, transmittance, dists, sdf_raw):
        obj_density = self.density(sdf_raw).transpose(0, 1).reshape(-1, dists.shape[0], dists.shape[1]) # [#object, #ray, #sample points]       
        free_energy = dists * obj_density
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        object_weight = alpha * transmittance
        return object_weight

    def get_color_prior_loss(self):

        list_idx = self.prior_idx_iter % (len(self.prior_obj_idx_list))
        sim_obj_idx = self.prior_obj_idx_list[list_idx]
        group_prior_obj_list = self.group_list[sim_obj_idx]

        if sim_obj_idx == 0:                                        # not use sds prior for bg
            self.prior_idx_iter += 1
            list_idx = self.prior_idx_iter % (len(self.prior_obj_idx_list))
            sim_obj_idx = self.prior_obj_idx_list[list_idx]
            group_prior_obj_list = self.group_list[sim_obj_idx]

        update_state = (sim_obj_idx == self.prior_obj_idx_list[-1])          # update after finish all object index

        self.prior_idx_iter += 1

        self.prior.set_prompt_renderer_idx(sim_obj_idx, group_prior_obj_list)

        self.mesh_color_network.set_obj_idx_list_bbox(sim_obj_idx)

        # use high resolution for color prior
        self.prior.camera_module.height = 128
        self.prior.camera_module.width = 128

        if 0 in group_prior_obj_list:           # prior with bg
            cam_loc_temp = torch.tensor([0, 0, 0])
            camera_batch = self.prior.get_scene_camera_views(camera_center=cam_loc_temp)
        else:
            camera_batch = self.prior.get_camera_views(update_camera=False)

        sds_loss_dict = self.prior.get_sds_loss(camera_batch, sim_obj_idx)

        if update_state:                                       # update after finish all object index
            self.prior.true_global_step += 1
            self.prior.global_step += 1

        return sds_loss_dict

    def get_scene_rgb(self, model_input, indices, height, width, render_bg=False, return_all=False):
        '''
        use color mesh to render scene rgb
        NOTE: replica use opengl coordinate system
              scannet, scannetpp use colmap coordinate system (opencv coordinate system)
              threestudio use colmap coordinate system
              so for replica dataset, we need to change the camera pose (opengl to colmap)
        '''
        
        print(f'render {indices.item()}th scene rgb, height: {height}, width: {width}')

        c2w = model_input["pose"].clone()                   # replica pose, in opengl coordinate system
        K = model_input["intrinsics"]               # different with proj_mtx in threestudio, proj_mtx is the the projection matrix in OpenGL
        skew = K[0, 0, 1]
        print(f'skew: {skew}')
        assert skew < 0.0001, 'skew should smaller than 0.001'

        # # NOTE: should use opengl perspective matrix, not camera intrinsics directly
        # # NOTE: get_projection_matrix from threestudio.utils.ops need cx = w/2, cy = h/2, not suitable for more general cases
        # # refer to https://github.com/threestudio-project/threestudio/issues/321
        # fy = K[0, 1, 1]                             # batch size = 1
        # fovy = 2 * torch.atan(height / (2 * fy))
        # fovy = torch.tensor([fovy.item()])
        # # proj_mtx = get_projection_matrix(fovy, width / height, 0.1, 1000)     # [1, 4, 4]

        proj_mtx = get_projection_matrix_my(K, width, height, 0.1, 1000)     # [1, 4, 4]
        proj_mtx = proj_mtx.cuda()

        # opengl to colmap
        c2w[:, :3, 1] *= -1
        c2w[:, :3, 2] *= -1
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)     # [B, 4, 4]

        camera_positions = c2w[:, :3, 3]
        pseudo_light_positions = torch.zeros_like(camera_positions)                         # not use light position
        pseudo_camera_distance = torch.norm(camera_positions, dim=-1, keepdim=True)         # not use depth

        if render_bg:
            mesh = self.prior.color_mesh_dict[0]
        else:
            mesh = self.prior.color_mesh_dict[1000]             # 1000 is the total scene mesh
        self.mesh_color_network.set_obj_idx_list_bbox(0)          # not convert coords

        # print('=======================')
        # print('mesh.v_pos:', mesh.v_pos.shape)
        # print('mvp_mtx:', mvp_mtx.shape)
        # print('c2w:', c2w.shape)
        # print('camera_positions:', camera_positions.shape)
        # print('pseudo_camera_distance:', pseudo_camera_distance.shape)
        # print('pseudo_light_positions:', pseudo_light_positions.shape)
        # print('height:', height)
        # print('width:', width)
        # print('=======================')
        out = self.prior.renderer(
                mesh, mvp_mtx=mvp_mtx, c2w=c2w, 
                camera_positions=camera_positions,
                camera_distances=pseudo_camera_distance,
                light_positions=pseudo_light_positions, 
                height=height, width=width, 
                render_rgb=True
              )


        self.prior._save_dir = os.path.join(self.prior._root_save_dir, f'input_views')
        os.makedirs(self.prior._save_dir, exist_ok=True)

        if return_all:
            return out

        return out['comp_rgb'][0]

    def infer_color_anchor_views(self, anchor_bin=10, elevation_range = [25, 35]):
        for sim_obj_idx, group_prior_obj_list in self.group_list.items():

            print(f'************** infer anchor range for object {sim_obj_idx} {group_prior_obj_list} **************')

            self.prior.set_prompt_renderer_idx(sim_obj_idx, group_prior_obj_list)

            self.mesh_color_network.set_obj_idx_list_bbox(sim_obj_idx)

            # use high resolution for color prior
            self.prior.camera_module.height = 384
            self.prior.camera_module.width = 384

            # get anchor camera views
            if 0 in group_prior_obj_list:          # prior with bg
                camera_batch = self.prior.get_scene_anchor_camera_views()
            else:
                camera_batch = self.prior.get_anchor_camera_views(anchor_bin, elevation_range, infer_color=True)

            if sim_obj_idx not in self.prior.color_mesh_dict:
                continue
            mesh = self.prior.color_mesh_dict[sim_obj_idx]

            total_out = None
            batch_num = camera_batch['azimuth'].shape[0]
            split_num = 8
            for idx in range(0, batch_num, split_num):
                camera_batch_split = {}
                for key in camera_batch:
                    if isinstance(camera_batch[key], torch.Tensor):
                        camera_batch_split[key] = camera_batch[key][idx:idx+split_num]
                    else:
                        camera_batch_split[key] = camera_batch[key]
                try:
                    out_split_temp = self.prior.renderer(mesh, **camera_batch_split)
                except Exception as e:
                    print(f'[ERROR]: {e}')
                    continue
                
                # detach for reduce memory
                out_split = {}
                out_split['comp_rgb'] = out_split_temp['comp_rgb'].detach()
                out_split['comp_normal_cam_white_vis'] = out_split_temp['comp_normal_cam_white_vis'].detach()
                del out_split_temp

                if total_out is None:
                    total_out = out_split
                else:
                    for key in total_out:
                        total_out[key] = torch.cat([total_out[key], out_split[key]], dim=0)

            self.prior._save_dir = os.path.join(self.prior._root_save_dir, 'infer-color-anchor-views', f'obj_{sim_obj_idx}')
            os.makedirs(self.prior._save_dir, exist_ok=True)

            # save image
            if total_out is None:
                print(f'[ERROR]: no output for object {sim_obj_idx}')
                continue
            for idx in range(total_out['comp_rgb'].shape[0]):

                azimuth = camera_batch['azimuth'][idx].item()
                elevation = camera_batch['elevation'][idx].item()
                
                self.prior.save_image_grid(
                    f"anchor_azimuth{azimuth:.2f}_elevation{elevation:.2f}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": total_out["comp_rgb"][idx],
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                        if "comp_rgb" in total_out
                        else []
                    )
                    + (
                        [
                            {
                                "type": "rgb",
                                "img": total_out["comp_normal_cam_white_vis"][idx],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                        if "comp_normal_cam_white_vis" in total_out
                        else []
                    ),
                    name="validation_step",
                    step=0,
                )

    def get_bg_pano_loss(self):

        sim_obj_idx = 0
        group_prior_obj_list = self.group_list[sim_obj_idx]

        self.prior.set_prompt_renderer_idx(sim_obj_idx, group_prior_obj_list)

        self.mesh_color_network.set_obj_idx_list_bbox(sim_obj_idx)

        # use high resolution for color prior
        self.prior.camera_module.height = 384
        self.prior.camera_module.width = 384

        # load total bg pano
        bg_anchor_root_path = os.path.join(self.prior._root_save_dir, f'obj_{sim_obj_idx}', 'bg_pano_anchor')
        cam_json_path = os.path.join(bg_anchor_root_path, 'bg_pano_anchor_camera.json')
        with open(cam_json_path, 'r') as f:
            cam_dict = json.load(f)
        color_map_path = os.path.join(bg_anchor_root_path, 'bg_pano_anchor_pers_gt.npy')
        bg_pano_gt_maps = np.load(color_map_path)

        # use 8 anchors
        batch_size = 8
        anchor_idx_list = np.random.choice(bg_pano_gt_maps.shape[0], size=batch_size, replace=False).tolist()

        camera_batch = {}
        camera_batch['mvp_mtx'] = torch.tensor(cam_dict['mvp_mtx'])[anchor_idx_list].cuda()
        camera_batch['camera_positions'] = torch.tensor(cam_dict['camera_positions'])[anchor_idx_list].cuda()
        camera_batch['c2w'] = torch.tensor(cam_dict['c2w'])[anchor_idx_list].cuda()
        camera_batch['camera_distances'] = torch.tensor(cam_dict['camera_distances'])[anchor_idx_list].cuda()
        camera_batch['camera_distances_relative'] = torch.tensor(cam_dict['camera_distances_relative'])[anchor_idx_list].cuda()
        camera_batch['light_positions'] = torch.zeros(batch_size, 3).cuda()
        camera_batch['height'] = 384
        camera_batch['width'] = 384

        pers_gt = bg_pano_gt_maps[anchor_idx_list]
        pers_gt = torch.from_numpy(pers_gt).float().cuda()
        
        try:
            mesh = self.prior.color_mesh_dict[sim_obj_idx]
            out = self.prior.renderer(mesh, **camera_batch)
            pred_rgb_map = out['comp_rgb']
        except Exception as e:
            print(f'[ERROR]: {e}')
            return torch.tensor(0.0).cuda()

        vis_threshold = self.bg_vis_thresh
        none_vis_mask = (out['vis_map'] < vis_threshold).float()
        pred_rgb_map = pred_rgb_map * none_vis_mask
        pers_gt = pers_gt * none_vis_mask

        bg_pano_loss = F.mse_loss(pred_rgb_map, pers_gt)

        self.prior._save_dir = os.path.join(self.prior._root_save_dir, f'obj_{sim_obj_idx}')
        os.makedirs(self.prior._save_dir, exist_ok=True)

        plot_interval = 100
        if self.prior.true_global_step % plot_interval == 0:

            for idx in range(out['comp_rgb'].shape[0]):
                
                self.prior.save_image_grid(
                    f"it{self.prior.true_global_step:06d}_{idx}.png",
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
                                "img": (out["vis_map"][idx] > vis_threshold).float(),
                                "kwargs": {"cmap": "jet"},
                            }
                        ]
                        if "vis_map" in out
                        else []
                    )
                    + (
                        [
                            {
                                "type": "rgb",
                                "img": pers_gt[idx],
                                "kwargs": {"data_format": "HWC"},
                            }
                        ]
                        if "vis_map" in out
                        else []
                    ),
                    name="validation_step",
                    step=self.prior.true_global_step,
                )

        return bg_pano_loss

    def bg_pano_anchor_views(self):

        from PIL import Image
        self.prior.camera_module.height = 384
        self.prior.camera_module.width = 384
        pano_height = self.prior.camera_module.height
        camera_batch = self.prior.get_scene_anchor_camera_views()

        sim_obj_idx = 0
        anchor_root_path = os.path.join(self.prior._root_save_dir, f'obj_{sim_obj_idx}', 'bg_pano_anchor')
        os.makedirs(anchor_root_path, exist_ok=True)

        # get pano anchor rgb map
        pano_root_path = os.path.join(self.prior._root_save_dir, f'obj_{sim_obj_idx}', 'bg_pano')
        # pano_root_path = './pano'
        pano_rgb_path = os.path.join(pano_root_path, 'bg_inpaint.png')
        pano_rgb_map = cv2.imread(pano_rgb_path)
        pano_rgb_map = cv2.cvtColor(pano_rgb_map, cv2.COLOR_BGR2RGB)
        pano_disp_path = os.path.join(pano_root_path, 'disp.npy')
        pano_disp = np.load(pano_disp_path)

        # colmap to opengl
        pano_cam_pose = camera_batch['c2w'].clone()
        pano_cam_pose[:, :3, 1] *= -1
        pano_cam_pose[:, :3, 2] *= -1

        fovy_deg = camera_batch['fovy']
        fovy = fovy_deg * torch.pi / 180
        fy = pano_height / (2 * torch.tan(fovy / 2))

        pers_gt_list = []
        for i in range(pano_cam_pose.shape[0]):
            intrinsics = torch.tensor([
                [fy[i], 0, pano_height/2], 
                [0, fy[i], pano_height/2], 
                [0, 0, 1]
            ])
            pers_img = pano_2_pers(pano_rgb_map, pano_disp, pano_cam_pose[i].cpu(), intrinsics.cpu(), w=pano_height, rot=0)
            
            # save pano image
            pers_img_map = Image.fromarray(pers_img)
            pers_img_map.save(os.path.join(anchor_root_path, f'pano_{i}.png'))
            
            pers_img = torch.from_numpy(pers_img).float().cuda() / 255
            pers_gt_list.append(pers_img)
        pers_gt = torch.stack(pers_gt_list, dim=0).cpu().numpy()

        anchor_dict = {}
        anchor_dict['mvp_mtx'] = camera_batch['mvp_mtx'].cpu().numpy().tolist()
        anchor_dict['camera_positions'] = camera_batch['camera_positions'].cpu().numpy().tolist()
        anchor_dict['c2w'] = camera_batch['c2w'].cpu().numpy().tolist()
        anchor_dict['azimuth'] = camera_batch['azimuth'].cpu().numpy().tolist()
        anchor_dict['elevation'] = camera_batch['elevation'].cpu().numpy().tolist()
        anchor_dict['camera_distances'] = camera_batch['camera_distances'].cpu().numpy().tolist()
        anchor_dict['camera_distances_relative'] = camera_batch['camera_distances_relative'].cpu().numpy().tolist()
        anchor_dict['height'] = camera_batch['height']
        anchor_dict['width'] = camera_batch['width']

        anchor_save_path = os.path.join(anchor_root_path, 'bg_pano_anchor_camera.json')
        with open(anchor_save_path, 'w') as f:
            json.dump(anchor_dict, f)

        # save pers gt
        pers_gt_save_path = os.path.join(anchor_root_path, 'bg_pano_anchor_pers_gt.npy')
        np.save(pers_gt_save_path, pers_gt)

        print('save anchor views to', anchor_root_path)
