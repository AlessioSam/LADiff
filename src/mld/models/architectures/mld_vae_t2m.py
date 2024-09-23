from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from mld.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask

from mld.models.architectures.encdec import Encoder, Decoder
"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""


class MldVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = ablation.MLP_DIST
        self.pe_type = ablation.PE_TYPE


        self.encoder = Encoder(input_emb_width=input_feats, down_t=3)
        self.decoder = Decoder(input_emb_width=input_feats, down_t=3)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x



    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        # input: [bs, nframes, nfeats]
        features = self.preprocess(features)
        #print("x_in", x_in.shape)
        x_encoder = self.encoder(features)  # (bs, 512, latent_dim)


        # match with mld_vae, added 19/maggio
        x_encoder = x_encoder.permute(2,0,1) # (latent_dim, bs, C)

        return x_encoder, None

    def decode(self, z: Tensor, lengths: List[int]):
        z = z.permute(1,2,0) # (bs, C, latent_dim)
        feats = self.decoder(z)
        feats = self.postprocess(feats)
        return feats
