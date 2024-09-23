# Stage 1
# python -m train --cfg configs/config_vae_humanml3d_edit.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug


# Stage 2
python -m train --cfg configs/config_remo_humanml3d_edit.yaml --cfg_assets configs/assets.yaml --batch_size 64  --nodebug