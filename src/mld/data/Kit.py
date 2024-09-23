import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import recover_from_ric

from .base import BASEDataModule
from .humanml.data.dataset import Text2MotionDatasetV2, TextOnlyDataset
from .utils import all_collate


class KitDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 phase='train',
                 collate_fn=all_collate,
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = 'kit'
        self.njoints = 21

        

        if phase == 'text_only':
            self.Dataset = TextOnlyDataset
        else:
            self.Dataset = Text2MotionDatasetV2
        self.cfg = cfg

        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        # self.transforms = self._sample_set.transforms

        self.their = True
        #! Added by me
        #if self.their==False: 
        self.mean  = np.load('./deps/t2m/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy') #! SOTA: QUESTO FUNZIONA 
        self.std = np.load('./deps/t2m/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')

        #! FROM REMO
        #self.mean = np.load('./datasets/kit-ml/mean_std_remo/mean.npy')
        #self.std = np.load('./datasets/kit-ml/mean_std_remo/std.npy')

    def feats2joints(self, features):
        #if self.their:
        #    mean = torch.tensor(self.hparams.mean).to(features)
        #    std = torch.tensor(self.hparams.std).to(features)
        #    features = features * std + mean
        #else: #! SOTA: QUESTO FUNZIONA 
        features = features * torch.tensor(self.std).to(features) + torch.tensor(self.mean).to(features)
        return recover_from_ric(features, self.njoints)

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        #! Like MLD
        if self.their: #! SOTA: QUESTO FUNZIONA 
            ori_mean = torch.tensor(self.hparams.mean).to(features)
            ori_std = torch.tensor(self.hparams.std).to(features)
        else:
            # Ours
            ori_mean = torch.tensor(self.mean).to(features)
            ori_std = torch.tensor(self.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)  
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
