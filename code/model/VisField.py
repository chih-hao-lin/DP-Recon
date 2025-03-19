import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import sys
sys.path.append(os.getcwd())
from model.embedder import *

class VisGrid(nn.Module):
    '''
    Dense grid, each point records importance sampling factors directly.
    NOTE: grid mapping total scene.
    '''
    def __init__(self, fill_data, resolution=256, divide_factor=1.1, device='cuda'):
        super().__init__()

        self.resolution = resolution
        self.divide_factor = divide_factor  # used to ensure the object bbox is in the grid
        
        # dense grid
        self.visgrid = nn.Parameter(torch.empty(1, 1, resolution, resolution, resolution, dtype=torch.float32, device=device))
        self.reset_parameters(fill_data)

        # create gaussian kernel
        self.kernel_size = 5
        self.sigma = 0.5
        self.kernel = self.gaussian_kernel(size=self.kernel_size, sigma=self.sigma, device=device)

    def reset_parameters(self, fill_data):
        
        self.visgrid.data.fill_(fill_data)

    def gaussian_kernel(self, size=5, sigma=0.5, device='cuda'):
        """
        creates gaussian kernel with side length size and a sigma of sigma
        """
        grid_x, grid_y, grid_z = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32, device=device) for _ in range(3)], indexing='ij'
        )
        grid_x = grid_x - size // 2
        grid_y = grid_y - size // 2
        grid_z = grid_z - size // 2
        sq_distances = grid_x ** 2 + grid_y ** 2 + grid_z ** 2
        kernel = torch.exp(-sq_distances / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # add two dimensions for 'batch' and 'channel'

    def update_visgrid(self):
        # update visgrid
        # NOTE: update visgrid after optimizer.step()
        # use gaussian filter to smooth visgrid
        self.visgrid.data = F.conv3d(self.visgrid.data, self.kernel, padding=self.kernel_size//2)

    def forward(self, points):
        '''
        points: [num_points, 3]
        '''
        num_points = points.shape[0]

        # ensure the object bbox is in the grid
        points = points / self.divide_factor

        # NOTE: transpose x, z
        points = points[:, [2, 1, 0]]                       # [num_points, 3]

        points = points.reshape(1, 1, 1, num_points, 3)     # [1, 1, 1, num_points, 3]

        sampling_factor = F.grid_sample(self.visgrid, points, align_corners=True)  # [1, 1, 1, 1, num_points]
        sampling_factor = sampling_factor.reshape(num_points, 1)      # [num_points, 1]

        return sampling_factor
    
    def get_loss(self, max_trans, pred_transmittance):
        '''
        sampling_factor: [num_points, 1]
        '''

        loss = torch.max(pred_transmittance.reshape(-1, 1).detach() - max_trans, torch.zeros_like(max_trans)).mean()

        return loss
    
    def grid_parameters(self):
        print("grid parameters", len(list(self.parameters())))
        for p in self.parameters():
            print(p.shape)
        return self.parameters()


class VisMLP(nn.Module):
    def __init__(self, multires, input_cam_poses, input_size=3, feature_size=256, hidden_size=512, output_size=1, num_hidden_layers=2):

        '''
        mlp input size: input_size + feature_size
        input_size: 3 (ray direction)
        feature_size: 256 (feature from radiance field)
        '''

        super(VisMLP, self).__init__()

        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_size = get_embedder(multires, input_dims=input_size)

        # Define the input layer
        self.input_layer = nn.Linear(input_size+feature_size, hidden_size)

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        ])

        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        input_cam_poses = torch.stack(input_cam_poses).cuda()
        self.input_cam_locs = input_cam_poses[:, :3, 3]

    def forward(self, ray_dirs, feature_vectors):

        '''
        ray_dirs: [M, 3], ray directions
        feature_vectors: [M, feature_size], feature from radiance field

        x: [M, 1], transmittance in [0, 1]
        '''

        if self.embed_fn is not None:
            ray_dirs = self.embed_fn(ray_dirs)
        x = torch.cat([ray_dirs, feature_vectors], dim=-1)

        # Forward pass through the input layer
        x = F.relu(self.input_layer(x))

        # Forward pass through the hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Forward pass through the output layer
        x = self.output_layer(x)
        x = F.sigmoid(x)                            # [0, 1], transmittance

        return x
    
    # def get_vis(self, src_points, feature_vectors):

    #     '''
    #     src_points: [num_points, 3], points in the src view for rendering in world coords, total num_points
    #     feature_vectors: [num_points, feature_size], feature from radiance field
    #     '''

    #     ray_dirs = self.get_new_ray_dirs(src_points)            # [num_points, N, 3]

    #     num_points, N, _ = ray_dirs.shape
    #     ray_dirs = ray_dirs.reshape(num_points*N, -1)           # [num_points*N, 3]

    #     feature_vectors = feature_vectors.unsqueeze(1).repeat(1, N, 1).reshape(num_points*N, -1)  # [num_points*N, feature_size]

    #     with torch.no_grad():       # no need to compute gradients
    #         raw_transmittance = self.forward(ray_dirs, feature_vectors)         # [num_points*N, 1]
    #         raw_transmittance = raw_transmittance.reshape(num_points, N)        # [num_points, N]

    #     # get the max transmittance for each point
    #     max_transmittance, _ = torch.max(raw_transmittance, dim=1)          # [num_points]
    #     max_transmittance = max_transmittance.unsqueeze(1)                  # [num_points, 1]

    #     return max_transmittance
    
    def get_vis(self, src_points, feature_vectors, max_order=5):

        '''
        src_points: [num_points, 3], points in the src view for rendering in world coords, total num_points
        feature_vectors: [num_points, feature_size], feature from radiance field
        '''

        ray_dirs = self.get_new_ray_dirs(src_points)            # [num_points, N, 3]

        num_points, N, _ = ray_dirs.shape
        ray_dirs = ray_dirs.reshape(num_points*N, -1)           # [num_points*N, 3]

        feature_vectors = feature_vectors.unsqueeze(1).repeat(1, N, 1).reshape(num_points*N, -1)  # [num_points*N, feature_size]

        with torch.no_grad():       # no need to compute gradients
            raw_transmittance = self.forward(ray_dirs, feature_vectors)         # [num_points*N, 1]
            raw_transmittance = raw_transmittance.reshape(num_points, N)        # [num_points, N]

        # get the max transmittance for each point
        top_max_transmittance, _ = raw_transmittance.topk(max_order, dim=1)     # [num_points, max_order]
        max_order_transmittance = top_max_transmittance[:, -1].unsqueeze(1)    # [num_points, 1], use the max order value

        return max_order_transmittance
    
    def get_new_ray_dirs(self, src_points):
        '''
        src_points: [num_points, 3], points in the src view for rendering in world coords, total num_points
        self.input_cam_locs: [N, 3], input views camera locations in world coords, total N views
        
        new_ray_dirs: [num_points, N, 3], new ray directions for each point in each input views
        '''
        num_points = src_points.shape[0]
        N = self.input_cam_locs.shape[0]

        src_points = src_points.unsqueeze(1).repeat(1, N, 1)  # [num_points, N, 3]
        input_cam_locs = self.input_cam_locs.unsqueeze(0).repeat(num_points, 1, 1)  # [num_points, N, 3]

        new_ray_dirs = src_points - input_cam_locs
        new_ray_dirs = F.normalize(new_ray_dirs, dim=2)

        return new_ray_dirs


if __name__ == '__main__':

    # test dense grid sampling
    densegrid_sampling = VisGrid(fill_data=0.0)
    # print(densegrid_sampling)

    optimizer = torch.optim.Adam(list(densegrid_sampling.parameters()), lr=0.01)

    # points = torch.rand(10, 3, dtype=torch.float32, device='cuda')
    points = torch.tensor(
        [
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.45, 0.45, 0.45],
            [0.55, 0.55, 0.55],
            [0.6, 0.6, 0.6],
            # [0.107, 0.107, 0.107],
            [0.114, 0.114, 0.114],
        ],
        dtype=torch.float32, device='cuda'
    )
    print(points)

    test_points = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [-0.2, -0.2, -0.2],
            [0.1, 0.1, 0.1],
        ],
        dtype=torch.float32, device='cuda'
    )

    for i in range(10):
        optimizer.zero_grad()
        sampling_factor = densegrid_sampling(points)
        # print(sampling_factor.shape)

        print('sampling_factor:', sampling_factor)

        test_sampling_factor = densegrid_sampling(test_points)
        print('test_sampling_factor: ', test_sampling_factor)

        loss = densegrid_sampling.get_loss(sampling_factor)
        loss.backward()
        optimizer.step()

        # add gaussian filter to smooth visgrid
        densegrid_sampling.update_visgrid()

