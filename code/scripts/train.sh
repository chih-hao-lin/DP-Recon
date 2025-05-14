# torchrun training/exp_runner.py --conf confs/replica_grid.conf  --scan_id 1 --prior_yaml geometry.yaml

# # Training first stage
# CUDA_LAUNCH_BLOCKING=1 torchrun training/exp_runner.py --conf confs/ours/replica.conf  --scan_id 1 --prior_yaml geometry.yaml \
#    --is_continue --ft_folder ../exps/dprecon_replica_1/2025_05_12_19_10_05 --checkpoint 110 # --force_init_visgrid

# # inpaint bg
# python ControlNetInpaint/inpaint_pano.py --root_dir ../exps/dprecon_grid_replica_0/2025_05_11_22_48_58/plots/sds_views/obj_0/bg_pano --save_dir ../exps/dprecon_grid_replica_0/2025_05_11_22_48_58/plots/sds_views/obj_0/bg_pano/inpaint --inpaint_repeat_times 1

# Manually select the best background inpainting result from '../exps/plots/sds_views/obj_0/bg_pano/inpaint/' to replace randomly selected ../exps/plots/sds_views/obj_0/bg_pano/bg_inpaint.png
# dir_exp=../exps/dprecon_replica_1/2025_05_13_13_27_42
# cp $dir_exp/plots/sds_views/obj_0/bg_pano/inpaint/rgb_map.png $dir_exp/plots/sds_views/obj_0/bg_pano/bg_inpaint.png

CUDA_LAUNCH_BLOCKING=1 torchrun training/exp_runner.py --conf confs/ours/replica.conf  --scan_id 1 --prior_yaml texture.yaml \
    --is_continue --ft_folder ../exps/dprecon_replica_1/2025_05_13_13_27_42 --checkpoint 225