import os
import json
import torch
import trimesh
from glob import glob
import shutil


def get_prior_bbox_withvis(visgrid, root_path, epoch, vis_thresh=0.1):

    bg_x_min, bg_x_max = -1.0, 1.0
    bg_y_min, bg_y_max = -1.0, 1.0
    bg_z_min, bg_z_max = -1.0, 1.0

    bbox_root_path = os.path.join(root_path, 'prior_bbox')
    os.makedirs(bbox_root_path, exist_ok=True)

    mesh_list = sorted(glob(os.path.join(root_path, f'surface_{epoch}_*.ply')))
    for mesh_path in mesh_list:
        if 'whole' in mesh_path:            # not for whole scene
            continue

        obj_idx = mesh_path.split('_')[-1].split('.')[0]

        if obj_idx == '0':
            bbox_json_path = os.path.join(bbox_root_path, f'bbox_{obj_idx}.json')
            obj_bbox = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
            with open(bbox_json_path, 'w') as f:
                json.dump(obj_bbox, f)
            print(f'use default bg bbox because bg has too many vertices, bbox_{obj_idx}.json save to {bbox_root_path}')
            continue

        bbox_json_path = os.path.join(bbox_root_path, f'bbox_{obj_idx}.json')

        mesh = trimesh.load(mesh_path)
        # get mesh components
        mesh_components = mesh.split(only_watertight=False)

        merge_mesh_list = []
        max_vertice_num = 0
        max_component = None
        for component in mesh_components:
            pnts = component.vertices
            pnts = torch.tensor(pnts, dtype=torch.float32, device='cuda')

            max_trans = visgrid(pnts).detach()
            mean_trans = torch.mean(max_trans)
            if mean_trans > vis_thresh:
                merge_mesh_list.append(component)

            if pnts.shape[0] > max_vertice_num:
                max_vertice_num = pnts.shape[0]
                max_component = component

        if len(merge_mesh_list) == 0:
            merge_mesh_list.append(max_component)

        max_component_vertice_num = max_component.vertices.shape[0]
        exist_max_component = False
        for component in merge_mesh_list:
            if component.vertices.shape[0] == max_component_vertice_num:
                exist_max_component = True
                break
        if not exist_max_component:
            # use the top 1/4 points to get the mean max trans
            pnts = max_component.vertices
            pnts = torch.tensor(pnts, dtype=torch.float32, device='cuda')

            max_trans = visgrid(pnts).detach()
            select_num = int(pnts.shape[0] / 4)
            top_select_max_trans, _ = torch.topk(max_trans.reshape(-1), select_num, largest=True)
            top_mean_trans = torch.mean(top_select_max_trans)
            if top_mean_trans > vis_thresh:
                merge_mesh_list.append(max_component)

        meshexport = trimesh.util.concatenate(merge_mesh_list)

        x_min, x_max = meshexport.vertices[:, 0].min() - 0.1, meshexport.vertices[:, 0].max() + 0.1
        y_min, y_max = meshexport.vertices[:, 1].min() - 0.1, meshexport.vertices[:, 1].max() + 0.1
        z_min, z_max = meshexport.vertices[:, 2].min() - 0.1, meshexport.vertices[:, 2].max() + 0.1
        x_min, x_max = max(x_min, bg_x_min), min(x_max, bg_x_max)
        y_min, y_max = max(y_min, bg_y_min), min(y_max, bg_y_max)
        z_min, z_max = max(z_min, bg_z_min), min(z_max, bg_z_max)
        obj_bbox = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        with open(bbox_json_path, 'w') as f:
            json.dump(obj_bbox, f)
        print(f'bbox_{obj_idx}.json save to {bbox_root_path}')

        meshexport.export(os.path.join(bbox_root_path, f'surface_{epoch}_{obj_idx}_filter.ply'))

    print('************ export prior bbox done ************')

    return bbox_root_path

def fix_object_prior_bbox(root_path, fix_location_json_path):
    '''
    fix heavily occluded objects bbox
    '''
    prior_bbox_root_path = os.path.join(root_path, 'prior_bbox')
    with open(fix_location_json_path, 'r') as f:
        fix_location_dict = json.load(f)

    ori_prior_bbox_root_path = os.path.join(prior_bbox_root_path, 'ori')
    os.makedirs(ori_prior_bbox_root_path, exist_ok=True)

    # get scene min z
    scene_min_z = 1.0                                 # default value, the max z of the whole scene
    for key in fix_location_dict.keys():
        if key == '0':
            continue

        bbox_json_path = os.path.join(prior_bbox_root_path, f'bbox_{key}.json')
        ori_bbox_json_path = os.path.join(ori_prior_bbox_root_path, f'bbox_{key}.json')
        shutil.copyfile(bbox_json_path, ori_bbox_json_path)
        
        support_by_floor = fix_location_dict[key]                       # 1 means support by floor, 0 means not
        if support_by_floor == 1:
            with open(bbox_json_path, 'r') as f:
                obj_bbox = json.load(f)

            obj_min_z = obj_bbox[0][2]
            scene_min_z = min(scene_min_z, obj_min_z)
    print(f'scene_min_z: {scene_min_z}')

    # fix obj bbox
    for key in fix_location_dict.keys():
        if key == '0':
            continue

        support_by_floor = fix_location_dict[key]
        if support_by_floor == 1:
            bbox_json_path = os.path.join(prior_bbox_root_path, f'bbox_{key}.json')
            with open(bbox_json_path, 'r') as f:
                obj_bbox = json.load(f)

            obj_bbox[0][2] = scene_min_z
            with open(bbox_json_path, 'w') as f:
                json.dump(obj_bbox, f)
    
    print('************ fix bbox done ************')

