#from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip
import random
import math

class RetrievalDatabase(nn.Module):

    def __init__(self,
                 num_retrieval=1,
                 topk=2,
                 retrieval_file='./ReMoDiffuse/data/database/t2m_text_train.npz',
                 latent_dim=512,
                 output_dim=512,
                 num_layers=2,
                 num_motion_layers=4,
                 kinematic_coef=0.1,
                 max_seq_len=196,
                 num_heads=8,
                 ff_size=1024,
                 stride=4,
                 sa_block_cfg=dict(
                type='EfficientSelfAttention',
                latent_dim=512,
                num_heads=8,
                dropout=0
            ),
                 ffn_cfg=dict(
                latent_dim=512,
                ffn_dim=1024,
                dropout=0,
            ),
                 dropout=0):
        super().__init__()
        self.num_retrieval = num_retrieval
        self.topk = topk
        self.latent_dim = latent_dim
        self.stride = stride
        self.kinematic_coef = kinematic_coef
        self.num_layers = num_layers
        self.num_motion_layers = num_motion_layers
        self.max_seq_len = max_seq_len
        data = np.load(retrieval_file)
        self.text_features = torch.Tensor(data['text_features'])
        self.captions = data['captions']
        self.motions = data['motions']
        self.m_lengths = data['m_lengths']
        self.clip_seq_features = data['clip_seq_features']
        self.train_indexes = data.get('train_indexes', None)
        self.test_indexes = data.get('test_indexes', None)

        only_retrieval=True
        if not only_retrieval:
            self.latent_dim = latent_dim
            self.output_dim = output_dim
            self.motion_proj = nn.Linear(self.motions.shape[-1], self.latent_dim)
            self.motion_pos_embedding = nn.Parameter(torch.randn(max_seq_len, self.latent_dim))
            self.motion_encoder_blocks = nn.ModuleList()
            for i in range(num_motion_layers):
                self.motion_encoder_blocks.append(
                    EncoderLayer(
                        sa_block_cfg=sa_block_cfg,
                        ffn_cfg=ffn_cfg
                    )
                )
            TransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation="gelu")
            self.text_encoder = nn.TransformerEncoder(
                TransEncoderLayer,
                num_layers=num_layers)
        self.results = {}

    def extract_text_feature(self, text, clip_model, device):
        text = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            #!text_features = clip_model.encode_text(text)
            text_features = clip_model(text)
        return text_features
    
    def encode_text(self, text, device):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        # B, T, D
        xf_out = x.permute(1, 0, 2)
        return xf_out

    def retrieve(self, caption, length, clip_model=None, device=None, idx=None):
        '''
        if self.training and self.train_indexes is not None and idx is not None:
            idx = idx.item()
            indexes = self.train_indexes[idx]
            data = []
            cnt = 0
            for retr_idx in indexes:
                if retr_idx != idx:
                    data.append(retr_idx)
                    cnt += 1
                    if cnt == self.topk:
                        break
            random.shuffle(data)
            return data[:self.num_retrieval]
        
        elif not self.training and self.test_indexes is not None and idx is not None:
            idx = idx.item()
            indexes = self.test_indexes[idx]
            data = []
            cnt = 0
            for retr_idx in indexes:
                data.append(retr_idx)
                cnt += 1
                if cnt == self.topk:
                    break
            # random.shuffle(data)
            return data[:self.num_retrieval]
        else:'''
        
        value = hash(caption)
        if value in self.results:
            return self.results[value]
        text_feature = self.extract_text_feature(caption, clip_model, device)
        text_feature = caption
        device = text_feature.device
        
        
        rel_length = torch.LongTensor(self.m_lengths).to(device)
        rel_length = torch.abs(rel_length - length) / torch.clamp(rel_length, min=length)
        semantic_score = F.cosine_similarity(self.text_features.to(device), text_feature)
        kinematic_score = torch.exp(-rel_length * self.kinematic_coef)
        score = semantic_score * kinematic_score
        indexes = torch.argsort(score, descending=True)
        data = []
        cnt = 0
        print('Indexes: ', indexes)
        quit()
        for idx in indexes:
            caption, motion, m_length = self.captions[idx], self.motions[idx], self.m_lengths[idx]
            #!if not self.training or m_length != length:
            if m_length != length:
                cnt += 1
                data.append(idx.item())
                if cnt == self.num_retrieval:
                    self.results[value] = data
                    return data
        #!assert False

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, captions, lengths, clip_model, device, idx=None):
        B = len(captions)
        all_indexes = []
        for b_ix in range(B):
            length = int(lengths[b_ix])
            if idx is None:
                batch_indexes = self.retrieve(captions[b_ix], length, clip_model, device)
            else:
                batch_indexes = self.retrieve(captions[b_ix], length, clip_model, device, idx[b_ix])
            all_indexes.extend(batch_indexes)
        all_indexes = np.array(all_indexes)
        N = all_indexes.shape[0]
        all_motions = torch.Tensor(self.motions[all_indexes]).to(device)
        all_m_lengths = torch.Tensor(self.m_lengths[all_indexes]).long()
        all_captions = self.captions[all_indexes].tolist()
            
        T = all_motions.shape[1]
        src_mask = self.generate_src_mask(T, all_m_lengths).to(device)
        raw_src_mask = src_mask.clone()
        re_motion = self.motion_proj(all_motions) + self.motion_pos_embedding.unsqueeze(0)
        for module in self.motion_encoder_blocks:
            re_motion = module(x=re_motion, src_mask=src_mask.unsqueeze(-1))
        re_motion = re_motion.view(B, self.num_retrieval, T, -1).contiguous()
        # stride
        re_motion = re_motion[:, :, ::self.stride, :].contiguous()
        
        src_mask = src_mask[:, ::self.stride].contiguous()
        src_mask = src_mask.view(B, self.num_retrieval, -1).contiguous()

        T = 77
        all_text_seq_features = torch.Tensor(self.clip_seq_features[all_indexes]).to(device)
        all_text_seq_features = all_text_seq_features.permute(1, 0, 2)
        re_text = self.text_encoder(all_text_seq_features)
        re_text = re_text.permute(1, 0, 2).view(B, self.num_retrieval, T, -1).contiguous()
        re_text = re_text[:, :, -1:, :].contiguous()
        
        # T = re_motion.shape[2]
        # re_feat = re_feat.view(B, self.num_retrieval * T, -1).contiguous()
        re_dict = dict(
            re_text=re_text,
            re_motion=re_motion,
            re_mask=src_mask,
            raw_motion=all_motions,
            raw_motion_length=all_m_lengths,
            raw_motion_mask=raw_src_mask)
        return re_dict
