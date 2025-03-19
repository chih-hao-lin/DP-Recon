import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import math
import cv2
import trimesh
import pyrender
from tqdm import tqdm
import argparse

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# hard-coded image size for replica dataset
H = 680
W = 1200

fx = 600.0
fy = 600.0
fx = 600.0
cx = 599.5
cy = 339.5
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    # import pdb; pdb.set_trace()

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def get_scene_anchor_camera(azimuth_bin=30, elevation_bin=30, camera_center=torch.zeros(3)):
    '''
    Generate camera pose in the scene
    '''
    azimuth_range = [-180, 180]
    elevation_range = [-90, 90]
    azimuth_repeat_times = (azimuth_range[1] - azimuth_range[0]) // azimuth_bin
    elevation_repeat_times = (elevation_range[1] - elevation_range[0]) // elevation_bin
    anchor_batch_size = azimuth_repeat_times * elevation_repeat_times

    azimuth_deg = torch.arange(-180, 180, azimuth_bin).float()
    azimuth_deg = azimuth_deg.repeat(elevation_repeat_times)
    azimuth = azimuth_deg * math.pi / 180

    elevation_deg = None
    for idx in range(elevation_repeat_times):
        sample_elevation_range = [elevation_bin*idx - 90, elevation_bin*(idx+1) - 90]
        sample_elevation_deg = (
            torch.rand(azimuth_repeat_times)
            * (sample_elevation_range[1] - sample_elevation_range[0])
            + sample_elevation_range[0]
        )
        if elevation_deg is None:
            elevation_deg = sample_elevation_deg
        else:
            elevation_deg = torch.cat((elevation_deg, sample_elevation_deg), dim=0)
    elevation = elevation_deg * math.pi / 180

    # camera position
    x_cam = (torch.rand(anchor_batch_size) * 2 - 1) * 0.05
    y_cam = (torch.rand(anchor_batch_size) * 2 - 1) * 0.05
    z_cam = (torch.rand(anchor_batch_size) * 2 - 1) * 0.02
    camera_positions = torch.stack([x_cam, y_cam, z_cam], dim=-1) + camera_center

    cam_radius = torch.rand(anchor_batch_size)

    center = torch.stack(
        [
            cam_radius * torch.cos(elevation) * torch.cos(azimuth), # x
            cam_radius * torch.cos(elevation) * torch.sin(azimuth), # y
            cam_radius * torch.sin(elevation), # z
        ],
        dim=-1,
    )

    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :
    ].repeat(anchor_batch_size, 1)

    # calculate up direction
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0

    return c2w

class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh, smooth = False)

    def delete(self):
        self.renderer.delete()

def refuse(mesh, poses):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for pose in tqdm(poses):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

parser = argparse.ArgumentParser(
    description='Arguments to cull the mesh.'
)
parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be culled')
parser.add_argument('--save_mesh', type=str,  help='path to the mesh to be culled')
parser.add_argument('--input_scalemat', type=str, help='path to the scale mat')
args = parser.parse_args()

# generate camera poses
cam_file = args.input_scalemat
camera_dict = np.load(cam_file)
total_views = len([k for k in camera_dict.files if 'scale_mat' in k])
scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(total_views)]
world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(total_views)]

tsdf_poses = []
for scale_mat, world_mat in zip(scale_mats, world_mats):
    P = world_mat @ scale_mat
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)

    cam_center = pose[:3, 3]
    tsdf_pose = get_scene_anchor_camera(azimuth_bin=30, elevation_bin=30, camera_center=torch.from_numpy(cam_center))
    tsdf_poses.append(tsdf_pose)

poses = torch.cat(tsdf_poses, dim=0)

input_mesh_path = args.input_mesh
mesh = trimesh.load_mesh(input_mesh_path)

filter_mesh = refuse(mesh, poses)
out_mesh_path = args.save_mesh.replace('.ply', '_temp.ply')
o3d.io.write_triangle_mesh(out_mesh_path, filter_mesh)
print(f"Filtered mesh saved at {out_mesh_path}")

# use the max component of the mesh
mesh = trimesh.load(out_mesh_path)
connected_components = mesh.split(only_watertight=False)
max_vertices = 0
largest_component = None
for component in connected_components:
    if len(component.vertices) > max_vertices:
        max_vertices = len(component.vertices)
        largest_component = component
largest_component_save_path = args.save_mesh
largest_component.export(largest_component_save_path)
print(f"Largest component saved at {largest_component_save_path}")
