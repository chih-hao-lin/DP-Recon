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


torchrun training/exp_runner.py --conf confs/replica_grid.conf  --scan_id 7 --prior_yaml geometry.yaml