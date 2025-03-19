import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import numpy as np
import torch
from utils.cameras.cameras import Cameras, CameraType
import cv2


def capture_pano_depth(mesh, img_h, rot, pano_cam_center):
    c2w = torch.tensor([[[np.cos(rot), 0., -np.sin(rot), pano_cam_center[0]],
                         [0., 1., 0., pano_cam_center[1]],
                         [np.sin(rot), 0., np.cos(rot), pano_cam_center[2]]]])

    equirectangular_camera = Cameras(
        cx=float(img_h),
        cy=0.5 * float(img_h),
        fx=float(img_h),
        fy=float(img_h),
        width=2 * img_h,
        height=img_h,
        camera_to_worlds=c2w,
        camera_type=CameraType.EQUIRECTANGULAR,
    )
    camera_ray_bundle = equirectangular_camera.generate_rays(camera_indices=0)

    rays_o = camera_ray_bundle.origins  # [h, w, 3]
    rays_d = camera_ray_bundle.directions  # [h, w, 3]
    rays_o = rays_o.view(-1, 3)  # [h*w, 3]
    rays_d = rays_d.view(-1, 3)  # [h*w, 3]

    points, index_ray, _ = RayMeshIntersector(
        mesh).intersects_location(rays_o, rays_d, multiple_hits=False)

    coords = np.array(list(np.ndindex(img_h, img_h*2))).reshape(img_h,
                                                                img_h*2, -1).reshape(-1, 2)  # .transpose(1,0,2).reshape(-1,2)
    depth = trimesh.util.diagonal_dot(
        points - rays_o[0].numpy(), rays_d[index_ray].numpy())
    pixel_ray = coords[index_ray]
    # no depth value set 0.1
    depthmap = np.full([img_h, img_h*2], 0.1)
    depthmap[pixel_ray[:, 0], pixel_ray[:, 1]] = depth

    disp = 1/depthmap
    disp[disp == 10] = 0

    mask = depthmap != 0.1
    mask = np.stack([mask]*3, axis=2)*255

    return depthmap, disp, mask, points, pixel_ray

def pano_2_pers(pano_img, pano_disp, new_poses, new_intr, w=1024, rot=0):
    new_w = w
    new_h = w

    height, width = pano_img.shape[:2]

    pixel_offset = 0.5
    # projection

    image_coords = torch.meshgrid(torch.arange(
        height), torch.arange(width), indexing="ij")
    image_coords = (torch.stack(image_coords, dim=-1) +
                    pixel_offset).float()  # stored as (y, x) coordinates

    s_x = image_coords[:, :, 1]*2*torch.pi/width - torch.pi
    s_y = -image_coords[:, :, 0]*torch.pi/height + torch.pi/2

    pano_depth = torch.from_numpy(1/pano_disp).float()
    w_x = pano_depth * torch.sin(s_x) * torch.cos(s_y)
    w_y = pano_depth * torch.cos(s_x) * torch.cos(s_y)
    w_z = pano_depth * torch.sin(s_y)

    pano_world_coords = torch.stack(
        (w_x.view(-1), w_y.view(-1), w_z.view(-1), torch.ones_like(w_x.view(-1))), dim=0)

    c2w_rot = torch.tensor([[[np.cos(rot), np.sin(rot), 0.0, 0.0],
                             [-np.sin(rot), np.cos(rot), 0.0, 0.0],
                             [0., 0., 1., 0],
                             [0.0000,  0.0000,  0.0000,  1.0000]
                             ]]).float()
    pano_world_coords = torch.inverse(c2w_rot) @ pano_world_coords

    camera_coords = torch.inverse(new_poses) @ pano_world_coords

    image_coords = new_intr @ camera_coords[:, 0:3, :]
    mask = image_coords[:, 2] > 1e-3
    z = image_coords[:, 2]
    z[z < 1e-3] = 1e-3

    u = (image_coords[:, 0]/z).view(-1)
    v = (image_coords[:, 1]/z).view(-1)

    mask = (u >= 0) & (u < (new_w-0.5)) & (v >=
                                           0) & (v < (new_h-0.5)) & mask.view(-1)
    mask = mask.bool()
    total_img_warp = np.zeros((new_h, new_w, 3))

    u_view = torch.round(u[mask]).int().numpy()
    v_view = torch.round(v[mask]).int().numpy()

    total_masked_img = (pano_img.reshape(-1, 3))[mask]
    z_depth = (z.view(-1))[mask]
    img_room_warp_depth = np.ones((new_h, new_w))*100
    for i in range(v_view.shape[0]):
        if img_room_warp_depth[v_view[i], u_view[i]] > z_depth[i]:
            total_img_warp[v_view[i], u_view[i]] = total_masked_img[i]
            img_room_warp_depth[v_view[i], u_view[i]] = z_depth[i]
        # if img_room_warp_depth[v_view[i], u_view[i]] > img_warp_depth[v_view[i], u_view[i]]:
        #         total_img_warp[v_view[i], u_view[i]] = img_warp[v_view[i], u_view[i]]
        #         img_room_warp_depth[v_view[i], u_view[i]] = img_warp_depth[v_view[i], u_view[i]]
    # show_img(total_img_warp.astype(np.uint8))
    new_re_img = cv2.inpaint(total_img_warp.astype(np.uint8), (img_room_warp_depth == 100).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return new_re_img.astype(np.uint8)


def get_depth(pose, mesh, resolution, f=500.0):
    scale = resolution/1024.0
    # transform pose in different coordinate system
    depth_pose = pose.clone()
    depth_pose[:, :, 1] = -depth_pose[:, :, 1]
    depth_pose[:, :, 2] = -depth_pose[:, :, 2]

    c2w = depth_pose[0, :3, :]

    width_pers = int(1024*scale)
    height_pers = int(1024*scale)

    cx = (width_pers-1)/2
    cy = (height_pers-1)/2
    fx = f*scale
    fy = f*scale

    perspective_camera = Cameras(
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        width=width_pers,
        height=height_pers,
        camera_to_worlds=c2w,
        camera_type=CameraType.PERSPECTIVE,
    )
    camera_ray_bundle = perspective_camera.generate_rays(camera_indices=0)

    rays_o = camera_ray_bundle.origins  # [h, w, 3]
    rays_d = camera_ray_bundle.directions  # [h, w, 3]
    rays_o = rays_o.view(-1, 3)  # [h*w, 3]
    rays_d = rays_d.view(-1, 3)  # [h*w, 3]

    points, index_ray, _ = RayMeshIntersector(
        mesh, False).intersects_location(rays_o, rays_d, multiple_hits=False)

    coords = np.array(list(np.ndindex(height_pers, width_pers))).reshape(
        height_pers, width_pers, -1).reshape(-1, 2)  # .transpose(1,0,2).reshape(-1,2)
    # depth = trimesh.util.diagonal_dot(points - rays_o[0].numpy(), rays_d[index_ray].numpy())
    c2w = torch.concat(
        (c2w, torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)), dim=0)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    depth = np.abs((torch.inverse(c2w).numpy() @ points.T)[2])
    # depth = (points @ rotation.numpy() + trans.numpy())[:,-1]
    pixel_ray = coords[index_ray]
    # 创建深度图矩阵，并进行对应赋值，没值的地方为nan，即空值
    depthmap_pers = np.full([height_pers, width_pers], 0.1)
    depthmap_pers[pixel_ray[:, 0], pixel_ray[:, 1]] = depth
    return depthmap_pers
