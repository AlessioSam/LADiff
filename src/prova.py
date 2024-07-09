import numpy as np

kit = np.load('./datasets/kit-ml/new_joint_vecs/00001.npy')
hml = np.load('./datasets/humanml3d/new_joint_vecs/000000.npy')

mean_kit = np.load('./datasets/kit-ml/Mean.npy')
std_kit = np.load('./datasets/kit-ml/Std.npy')

mean_hml = np.load('./datasets/humanml3d/Mean.npy')
std_hml = np.load('./datasets/humanml3d/Std.npy')

#mean_ref = np.load('Mean.npy')

# CMP
mean_kit_meta= np.load('./deps/t2m/kit/Comp_v6_KLD005/meta/mean.npy')
#std_kit = np.load('./deps/t2m/kit/Comp_v6_KLD005/meta/std.npy')

mean_hml_meta = np.load('./deps/t2m/t2m/Comp_v6_KLD01/meta/mean.npy')
#std_hml = np.load('./deps/t2m/t2m/Comp_v6_KLD01/meta/std.npy')

mean_t2mgpt_kit = np.load('./VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
mean_t2mgpt_hml = np.load('./VQVAEV3_CB1024_CMT_H1024_NRES3_hml/meta/mean.npy')

print(abs(mean_hml_meta - mean_t2mgpt_hml).mean())
