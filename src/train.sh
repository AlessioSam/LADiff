# Stage 1
# python -m train --cfg configs/config_vae_humanml3d_edit.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug


# Stage 2
python -m train --cfg configs/config_mld_humanml3d_edit.yaml --cfg_assets configs/assets.yaml --batch_size 128  --nodebug