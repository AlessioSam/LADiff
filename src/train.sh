# Stage 1
# python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug


# Stage 2
python -m train --cfg configs/config_ladiff_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 128  --nodebug