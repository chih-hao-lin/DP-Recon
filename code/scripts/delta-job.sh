#!/bin/sh
#
#SBATCH --job-name=replica_7
#SBATCH --output=/work/hdd/bcrp/cl121/DP-Recon/logs/%j.out
#SBATCH --error=/work/hdd/bcrp/cl121/DP-Recon/logs/%j.err
#
#SBATCH --account=bcrp-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=2-0:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

cd /work/hdd/bcrp/cl121/DP-Recon/code


# replica 
# CUDA_LAUNCH_BLOCKING=1 torchrun training/exp_runner.py --conf confs/ours/replica.conf   --scan_id 0 --prior_yaml geometry.yaml 

# scannetpp
# CUDA_LAUNCH_BLOCKING=1 torchrun training/exp_runner.py --conf confs/ours/scannetpp.conf --scan_id 0 --prior_yaml geometry.yaml 

# igibson
# CUDA_LAUNCH_BLOCKING=1 torchrun training/exp_runner.py --conf confs/ours/igibson.conf   --scan_id 0 --prior_yaml geometry.yaml 