import os
import numpy as np
import cv2
from PIL import Image
import openai
import base64
import io
import re
import json 
from glob import glob
from tqdm import tqdm


def read_api_key(path="/hdd/misc_openai_api_key.txt"):
    with open(path, "r") as f:
        return f.read().strip()

openai.api_key = read_api_key()
    
def img_path_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def img_array_to_base64(image_array):
    pil_image = Image.fromarray(image_array.astype('uint8'))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def vlm_inference(image_base64, prompt, model="gpt-4-turbo", max_tokens=300, seed=42):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful computer vision assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ],
        max_tokens=max_tokens,
        seed=seed
    )
    return response.choices[0].message.content
    
def label_with_vlm():
    dir_root = "/hdd/indoor_digital_twin/DP-Recon/data/replica/room_0"
    data_type = "replica"
    scan_id = 0
    dir_images = os.path.join(dir_root, "images")
    dir_mask = os.path.join(dir_root, "instance_mask")

    path_images = sorted(glob(os.path.join(dir_images, "*.jpg")))
    path_mask = sorted(glob(os.path.join(dir_mask, "*.png")))
    n_images = len(path_images)
    print(f"Number of images: {n_images}")

    instances = {}
    for i in tqdm(range(n_images)):
        mask_path = path_mask[i]

        # Read the image and mask
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask.astype(np.uint8)
        mask_ids = np.unique(mask)
        for mask_id in mask_ids:
            if mask_id == 255:
                continue
            instance_mask = (mask == mask_id)
            instance_area = np.sum(instance_mask)
            if mask_id not in instances:
                instances[mask_id] = {
                    "area": instance_area,
                    "image_id": i,
                }
            elif instance_area > instances[mask_id]["area"]:
                instances[mask_id]["area"] = instance_area
                instances[mask_id]["image_id"] = i

    instance_ids = list(instances.keys())
    instance_ids.sort()
    name_list = []
    touch_floor_list = []
    geometry_list = []
    appearance_list = []

    for instance_id in tqdm(instance_ids):
        image_id = instances[instance_id]["image_id"]
        image_path = path_images[image_id]
        mask_path = path_mask[image_id]

        # Read the image and mask
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask.astype(np.uint8)
        instance_mask = (mask == instance_id)
        obj_image = image.copy()
        obj_image[~instance_mask] = 0 # black background
        image_merged = np.concatenate((obj_image, image), axis=1)
        image_base64 = img_array_to_base64(image_merged)

        print('------------------------------')
        prompt_name = "Left is the target object with black background, and right is the full image. What is the object in the image? The object is in a room. Please ignore the black background. Please answer with a single word or a short phrase."
        name = vlm_inference(image_base64, prompt_name)
        name_list.append(name)
        print('name:', name)
        geometry = f"A {name}"
        geometry_list.append(geometry)
            
        prompt_touch_floor = "Left is the target object with black background, and right is the full image. Does the object touch the floor of the room? Please answer with yes or no."
        touch_floor = vlm_inference(image_base64, prompt_touch_floor).lower()
        if 'yes' in touch_floor:
            touch_floor_list.append(1)
        else:
            touch_floor_list.append(0)
        print('touch_floor:', touch_floor)

        appearance_prompt = "Left is the target object with black background, and right is the full image. Please describe the texture of the object in 20 words. Please ignore the black background."
        appearance = vlm_inference(image_base64, appearance_prompt)
        appearance_list.append(appearance)
        print('appearance:', appearance)
        
    instance_ids = np.array(instance_ids) + 1 # start from 1
    n_instance = len(instance_ids)
    name_list_organized = []
    for i, name in enumerate(name_list):
        name = name.replace(" ", "_")
        name = f"{name}_{i}"
        name_list_organized.append(name)
    instance_id_dict = {name_list_organized[i]: int(instance_ids[i]) for i in range(n_instance)}
    json_path = os.path.join(dir_root, "instance_id.json")
    with open(json_path, "w") as f:
        json.dump(instance_id_dict, f, indent=4)
    touch_floor_dict = {str(i+1): int(touch_floor_list[i]) for i in range(n_instance)}
    json_path = os.path.join(dir_root, f'{data_type}_scan{scan_id}_location_modify.json')
    with open(json_path, "w") as f:
        json.dump(touch_floor_dict, f, indent=4)

    room_geometry = "Empty room with smooth walls, floors and ceilings"
    room_appearance = "An isolated real estate photography style of wall and floor, matte surface, minimal detail, dark, fully unlit environment"
    geometry_list = np.insert(geometry_list, 0, room_geometry)
    appearance_list = np.insert(appearance_list, 0, room_appearance)

    geometry_dict = {str(i): geometry_list[i] for i in range(n_instance+1)}
    json_path = os.path.join(dir_root, f'{data_type}_scan{scan_id}_prompt_geometry.json')
    with open(json_path, "w") as f:
        json.dump(geometry_dict, f, indent=4)
    appearance_dict = {str(i): appearance_list[i] for i in range(n_instance+1)}
    json_path = os.path.join(dir_root, f'{data_type}_scan{scan_id}_prompt_texture.json')
    with open(json_path, "w") as f:
        json.dump(appearance_dict, f, indent=4)
    print("Labeling completed.")

def sample_eval_data():
    dir_root = "/hdd/indoor_digital_twin/DP-Recon/data/replica/room_0"
    data_type = "replica"
    scan_id = 0
    test_split_ratio=0.1
    first_k = 10
    dir_images = os.path.join(dir_root, "images")
    dir_mask = os.path.join(dir_root, "instance_mask")
    
    dir_eval = os.path.join(dir_root, "eval-novel-views")
    os.makedirs(dir_eval, exist_ok=True)
    dir_eval_rgb = os.path.join(dir_eval, "rgb")
    os.makedirs(dir_eval_rgb, exist_ok=True)
    dir_eval_mask = os.path.join(dir_eval, "instance_mask")
    os.makedirs(dir_eval_mask, exist_ok=True)

    image_paths = sorted(glob(os.path.join(dir_images, "*.jpg")))
    mask_paths  = sorted(glob(os.path.join(dir_mask, "*.png")))
    n_images = len(image_paths)
    num_test_split = int(n_images * test_split_ratio)
    train_split_indices = np.linspace(0, n_images - 1, n_images - num_test_split).astype(np.int32)
    test_split_indices = np.setdiff1d(np.arange(n_images), train_split_indices)
    test_split_indices = test_split_indices[:first_k].tolist()

    # copy RGB and mask
    for i in range(len(test_split_indices)):
        idx = test_split_indices[i]
        image_path_src = image_paths[idx]
        image_path_dst = os.path.join(dir_eval_rgb, f"{i:0>6}_rgb.png")
        os.system(f"cp {image_path_src} {image_path_dst}")

        mask_path_src = mask_paths[idx]
        mask_path_dst = os.path.join(dir_eval_mask, f"{i:0>6}.png")
        os.system(f"cp {mask_path_src} {mask_path_dst}")
    
    eval_dict = [i for i in range(len(test_split_indices))]
    eval_dict_path = os.path.join(dir_eval, "eval-10views.json")
    with open(eval_dict_path, "w") as f:
        json.dump(eval_dict, f, indent=4)

    camera_info_path = os.path.join(dir_root, "transforms.json")
    with open(camera_info_path, 'r') as f:
        camera_info = json.load(f)

    fx = camera_info['fl_x']
    fy = camera_info['fl_y']
    cx = camera_info['cx']
    cy = camera_info['cy']

    intrinsics = np.eye(4)
    intrinsics[:3, :3] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    t_all = []
    for i in range(len(camera_info['frames'])):
        pose = np.array(camera_info['frames'][i]['transform_matrix']).reshape(4, 4)
        t = pose[:3, 3]
        t_all.append(t)
    t_all = np.array(t_all)
    t_max = np.max(t_all, axis=0)
    t_min = np.min(t_all, axis=0)
    norm_scale = np.max(t_max - t_min)
    norm_center = 0.5 * (t_max + t_min)

    scale_mat = np.eye(4)
    cameras = {}
    for i in range(len(test_split_indices)):
        idx = test_split_indices[i]
        pose = np.array(camera_info['frames'][idx]['transform_matrix']).reshape(4, 4)
        # opengl to opencv
        pose[:, 1:3] *= -1
        t = pose[:3, 3]
        t_norm = (t - norm_center) / norm_scale
        pose[:3, 3] = t_norm
        world_mat = intrinsics @ pose
        cameras[f"scale_mat_{i}"] = scale_mat
        cameras[f"world_mat_{i}"] = world_mat
    cameras_path = os.path.join(dir_eval, "cameras.npz")
    np.savez(cameras_path, **cameras)
    print("Sample evaluation data completed.")

if __name__ == "__main__":
    # label_with_vlm()
    sample_eval_data()