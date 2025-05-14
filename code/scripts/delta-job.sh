#!/bin/sh
#
#SBATCH --job-name=igibson_1
#SBATCH --output=/work/hdd/bcrp/cl121/DP-Recon/logs/%j.out
#SBATCH --error=/work/hdd/bcrp/cl121/DP-Recon/logs/%j.err
#
#SBATCH --account=bcqn-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=2-0:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

cd /work/hdd/bcrp/cl121/DP-Recon/code


# replica 
# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/replica.conf  --scan_id 0 --prior_yaml geometry.yaml 
#    --is_continue --ft_folder ../exps/dprecon_replica_2/2025_05_12_23_22_37 --checkpoint 60

# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/replica.conf  --scan_id 0 --prior_yaml texture.yaml \
#     --is_continue --ft_folder ../exps/dprecon_replica_0/2025_05_13_16_06_22 --checkpoint 103

# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/replica.conf  --scan_id 1 --prior_yaml texture.yaml \
#     --is_continue --ft_folder ../exps/dprecon_replica_1/2025_05_13_14_36_16 --checkpoint 225

# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/replica.conf  --scan_id 2 --prior_yaml texture.yaml \
#     --is_continue --ft_folder ../exps/dprecon_replica_2/2025_05_13_14_36_16 --checkpoint 114


# scannetpp
# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/scannetpp.conf --scan_id 1 --prior_yaml geometry.yaml \
#     --is_continue --ft_folder ../exps/dprecon_scannetpp_1/2025_05_13_14_36_50 --checkpoint 220

# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/scannetpp.conf --scan_id 0 --prior_yaml texture.yaml \
#     --is_continue --ft_folder ../exps/dprecon_scannetpp_0/2025_05_13_14_36_39 --checkpoint 447

# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/scannetpp.conf --scan_id 1 --prior_yaml texture.yaml \
#     --is_continue --ft_folder ../exps/dprecon_scannetpp_1/2025_05_13_20_59_20 --checkpoint 220

# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/scannetpp.conf --scan_id 2 --prior_yaml texture.yaml \
#     --is_continue --ft_folder ../exps/dprecon_scannetpp_2/2025_05_13_16_38_03 --checkpoint 158

# igibson
# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/igibson.conf   --scan_id 1 --prior_yaml geometry.yaml \
#     --is_continue --ft_folder ../exps/dprecon_igibson_1/2025_05_12_19_01_26 --checkpoint 30

# CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/igibson.conf   --scan_id 0 --prior_yaml texture.yaml \
#     --is_continue --ft_folder ../exps/dprecon_igibson_0/2025_05_13_16_39_41 --checkpoint 46

CUDA_LAUNCH_BLOCKING=1 python training/exp_runner.py --conf confs/ours/igibson.conf   --scan_id 1 --prior_yaml texture.yaml \
    --is_continue --ft_folder ../exps/dprecon_igibson_1/2025_05_13_16_40_32 --checkpoint 59