# Stage 1
# python -m train --cfg configs/config_vae_humanml3d_vq.yaml --cfg_assets configs/assets.yaml --batch_size 512 --nodebug
python -m train --cfg configs/config_vae_humanact_edit.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug


# Stage 2
#python -m train --cfg configs/config_mld_kit_edit.yaml --cfg_assets configs/assets.yaml --batch_size 128  --nodebug



# ardiff_v0: full conditioning
# ardiff_v1: last conditioning
# ardiff_v2: conditioning full, both past and future

# ardiff_v1.1: conditioning last, mld stage1 [7,256], 

# ardiff_v4: conditioning last, mld stage1 vae_7latent_DVAE [7,256], DVAE = False at stage2
# ardiff_v4.1: (NOT RUNNED) conditioning last, mld stage1 vae_7latent_DVAE [7,256], DVAE = True at stage2 
# ardiff_v4_200: same as v4 but with 200 train timesteps for scheduler
# ardiff_v4_500: same as v4 but with 500 train timesteps for scheduler

# ardiff_v5: force seq

# ardiff_v6: (partially) fixed 8 frames handling + diffusion also sees cond_z=None