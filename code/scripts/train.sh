# torchrun training/exp_runner.py --conf confs/replica_grid.conf  --scan_id 1 --prior_yaml geometry.yaml

# Training first stage
# CUDA_LAUNCH_BLOCKING=1 torchrun training/exp_runner.py --conf confs/replica_ours.conf  --scan_id 0 --prior_yaml geometry.yaml \
#    --is_continue --ft_folder /hdd/indoor_digital_twin/DP-Recon/exps/dprecon_grid_replica_0/2025_05_11_15_28_47 --checkpoint 100 # --force_init_visgrid

# # inpaint bg
# python ControlNetInpaint/inpaint_pano.py --root_dir ../exps/dprecon_grid_replica_0/2025_05_11_22_48_58/plots/sds_views/obj_0/bg_pano --save_dir ../exps/dprecon_grid_replica_0/2025_05_11_22_48_58/plots/sds_views/obj_0/bg_pano/inpaint --inpaint_repeat_times 1

CUDA_LAUNCH_BLOCKING=1 torchrun training/exp_runner.py --conf confs/replica_ours.conf  --scan_id 0 --prior_yaml texture.yaml \
    --is_continue --ft_folder /hdd/indoor_digital_twin/DP-Recon/exps/dprecon_grid_replica_0/2025_05_11_22_48_58 --checkpoint 100