import os
import json
import argparse


def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split(': ')
            result_dict[key] = float(value)
    return result_dict


parser = argparse.ArgumentParser(
    description='Arguments to evaluate each object reconstruction metrics.'
)
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--root_path', type=str, default='./')
args = parser.parse_args()

dataset_type = args.dataset_type
root_path = args.root_path
exp_root_path = os.path.join(root_path, 'exps')
data_root_path = os.path.join(root_path, f'data/{dataset_type}')

total_results = {}          # for all scenes
total_scene = 0

exp_list = os.listdir(exp_root_path)
exp_list.sort()
for exp_name in exp_list:
    if f'_{dataset_type}_' not in exp_name:
        continue

    scan_id = exp_name.split('_')[-1]
    temp_exp_path = os.path.join(exp_root_path, exp_name)
    run_exp_list = os.listdir(temp_exp_path)
    run_exp_list.sort()
    run_exp_name = run_exp_list[-1]             # use the latest exp
    exp_path = os.path.join(temp_exp_path, run_exp_name, 'plots')

    save_obj_path = os.path.join(exp_path, 'eval_obj')
    os.makedirs(save_obj_path, exist_ok=True)

    print(f'Processing {exp_name}...')

    results = {}            # for single scene
    total_object = 0
    save_metrics_path = os.path.join(save_obj_path, 'metrics.txt')

    mesh_list = os.listdir(exp_path)
    mesh_list = [x for x in mesh_list if 'surface_' in x]
    epoch_list = [int(x.split('_')[1]) for x in mesh_list]
    max_epoch = max(epoch_list)
    eval_obj_list = [x for x in mesh_list if f'surface_{max_epoch}_' in x]

    for mesh_name in eval_obj_list:

        if 'whole' in mesh_name:            # skip total scene
            continue

        obj_id = int(mesh_name.split('_')[-1].split('.')[0])
        if obj_id == 0:                 # skip bg
            continue

        input_mesh = os.path.join(exp_path, mesh_name)
        gt_mesh = os.path.join(data_root_path, 'GTmesh', f'scan{scan_id}', f'obj_{obj_id}.ply')
        output_mesh = os.path.join(save_obj_path, f'fuse_obj_{obj_id}.ply')
        output_txt = os.path.join(save_obj_path, f'obj_{obj_id}_metrics.txt')

        if dataset_type == 'replica':
            input_scalemat = os.path.join(data_root_path, f'scan{scan_id}', 'cameras.npz')
        else:
            input_scalemat = os.path.join(data_root_path, f'scan{scan_id}', 'eval_total_cameras.npz')
        cmd = f'python eval/evaluate_single_scene.py --input_mesh {input_mesh} --gt_mesh {gt_mesh} --input_scalemat {input_scalemat} --output_mesh {output_mesh} --output_txt {output_txt} --none_fuse'

        print(cmd)
        os.system(cmd)

        metrics = read_txt_to_dict(output_txt)
        if not (metrics['F-score'] > 0):                    # skip invalid object
            obj_name = os.path.basename(output_txt)
            print(f'{exp_name} {obj_name} F-score: {metrics["F-score"]}')
            continue

        for key, value in metrics.items():
            if key not in results:
                results[key] = value
            else:
                results[key] += value
        
        total_object += 1

    save_metrics = {}
    print(f'scan{scan_id}, total object: {total_object}')
    for key, value in results.items():
        print(f'{key}: {value / total_object}')
        save_metrics[key] = value / total_object

    with open(save_metrics_path, 'w') as f:
        for k, v in save_metrics.items():
            out = f"{k}: {v}\n"
            f.write(out)

    # save total results
    for key, value in save_metrics.items():
        if key not in total_results:
            total_results[key] = value
        else:
            total_results[key] += value
    total_scene += 1

print(f'dataset: {dataset_type}, total objects: {total_scene}')
for key, value in total_results.items():
    print(f'{key}: {value / total_scene}')

total_object_results_txt_path = os.path.join(exp_root_path, f'total_obj_metrics_{dataset_type}.txt')
with open(total_object_results_txt_path, 'w') as f:
    for k, v in total_results.items():
        out = f"{k}: {v / total_scene}\n"
        f.write(out)

