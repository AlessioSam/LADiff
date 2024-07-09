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
        self.dvae = ablation.DVAE
        self.percentage_noised = ablation.PERCENTAGE_NOISED
        self.max_it = ablation.MAX_IT
        self.frame_per_latent = ablation.FRAME_PER_LATENT
        self.joint_distro_fix = ablation.JOINT_DISTRO_FIX
        self.LAD = ablation.LAD
        self.test_efficiency = ablation.TEST_EFFICIENCY

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(
                self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(
                self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder": # this
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            if self.max_it==0:
                self.global_motion_token = nn.Parameter(
                    torch.randn(self.latent_size * 2, self.latent_dim))
            else:
                self.global_motion_token = nn.Parameter(
                    torch.randn(self.max_it * 2, self.latent_dim))

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist
    

    def add_noise(self, x):
        x_feats = x.reshape(x.shape[0], -1).to(x.device)
        x_total = x_feats.shape[1]

        noise = torch.zeros_like(x_feats, device=x.device)

        
        index_to_corrupt = np.random.choice(x_total, int(x_total * self.percentage_noised)) # 0.5 0.33
        # Corrupt the features with gaussian noise for the selected indices
        noise[:, index_to_corrupt] = torch.randn_like(noise[:, index_to_corrupt])

        noise = noise.reshape(x.shape[0], x.shape[1], x.shape[2])
    
        x = x + noise
        return x
    
    def dist_to_mask(self, max_iter_elements, z):
        max_iter = z.shape[0] #self.max_it
        dist_masks = torch.ones((len(max_iter_elements), max_iter),
                                dtype=bool,
                                device=z.device)
        for i, max_iter_element in enumerate(max_iter_elements):
            dist_masks[i, max_iter_element:] = False
        return dist_masks


    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        # input: [bs, nframes, nfeats]
        device = features.device

        bs, nframes, nfeats = features.shape

        if self.dvae:
            features = self.add_noise(features)

        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1)) # max_it*2, bs, latent_dim
       
        
        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device) # bs, max_it*2
        
        if self.max_it!=0:
            max_iter_elements = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long) # (torch.tensor(lengths) // self.frame_per_latent + 1)
            if self.LAD:
                #max_len_batch = max(lengths)
                
                # split the dist mask into 2, each with self.max_it
                dist_masks_mu, dist_masks_logvar = torch.split(dist_masks, self.max_it, dim=1)
                for i, max_iter_element in enumerate(max_iter_elements):
                    dist_masks_mu[i, max_iter_element:] = False
                    dist_masks_logvar[i, max_iter_element:] = False
                dist_masks = torch.cat((dist_masks_mu, dist_masks_logvar), dim=1)

        aug_mask = torch.cat((dist_masks, mask), 1) # bs, max_it*2 + nframes

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0) 

        if self.pe_type == "actor":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        elif self.pe_type == "mld":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
            # query_pos = self.query_pos_encoder(xseq)
            # dist = self.encoder(xseq, pos=query_pos, src_key_padding_mask=~aug_mask)[
            #     : dist.shape[0]
            # ]

        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.joint_distro_fix:
            latent = torch.tensor([], device=dist.device)
            dist_list = []
            for i, max_iter_element in enumerate(max_iter_elements):
                mu = dist[0:max_iter_element, i]
                logvar = dist[self.max_it:self.max_it+max_iter_element, i]
                std = logvar.exp().pow(0.5)
                dist_ = torch.distributions.Normal(mu, std)
                dist_list.append(dist_)
                latent_ = dist_.rsample()
                if max_iter_element<self.max_it:
                    to_cat = torch.zeros((self.max_it-max_iter_element, self.latent_dim), device=dist.device)
                    latent_ = torch.cat((latent_, to_cat), dim=0).unsqueeze(1)
                else:
                    latent_ = latent_.unsqueeze(1)
                latent = torch.cat((latent, latent_), dim=1) if i!=0 else latent_
            dist = dist_list
          

        else:
            if self.mlp_dist:
                tokens_dist = self.dist_layer(dist)
                mu = tokens_dist[:, :, :self.latent_dim]
                logvar = tokens_dist[:, :, self.latent_dim:]
            else:
                if self.max_it==0:
                    mu = dist[0:self.latent_size, ...] # mu.shape [1, bs, 256]
                    logvar = dist[self.latent_size:, ...]
                else:
                    mu = dist[0:self.max_it, ...] # mu.shape [max_it, bs, 256]
                    logvar = dist[self.max_it:, ...]

            # resampling and padding output
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu, std)
            latent = dist.rsample()
            if self.max_it!=0:
                if self.LAD:
                    for i, max_iter_element in enumerate(max_iter_elements):
                        latent[max_iter_element:, i] = 0

        # if self.dvae:
        #     for i, max_iter_element in enumerate(max_iter_elements):
        #         feats = latent[:max_iter_element, i, :].reshape(1, -1)
        #         feats_total = feats.shape[1]

        #         noise = torch.zeros_like(feats, device=latent.device)

                
        #         index_to_corrupt = np.random.choice(feats_total, int(feats_total * self.percentage_noised))
        #         # Corrupt the features with gaussian noise for the selected indices
        #         noise[:, index_to_corrupt] = torch.randn_like(noise[:, index_to_corrupt])

        #         noise = noise.reshape(max_iter_element, latent.shape[-1])

        #         latent[:max_iter_element, i, :] = latent[:max_iter_element, i, :] + noise

        return latent, dist, max_iter_elements

    def decode(self, z: Tensor, lengths: List[int], plot_att_map=None, latentwise_gen=None):
        
        mask = lengths_to_mask(lengths, z.device) # [len(lengths), max(lengths)] with True if frame to consider else False

        max_iter_elements = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long) # (torch.tensor(lengths) // self.frame_per_latent + 1)
        
        if not self.test_efficiency:
            latent_mask = self.dist_to_mask(torch.tensor(range(1, self.max_it+1)) if latentwise_gen=="fw" else max_iter_elements, z)
        
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size),
                                dtype=bool,
                                device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.pe_type == "actor":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
            elif self.pe_type == "mld":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
                # query_pos = self.query_pos_decoder(xseq)
                # output = self.decoder(
                #     xseq, pos=query_pos, src_key_padding_mask=~augmask
                # )[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            if self.pe_type == "actor":
                queries = self.query_pos_decoder(queries)
                output = self.decoder(tgt=queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask,
                                      ).squeeze(0)
            elif self.pe_type == "mld":
                queries = self.query_pos_decoder(queries) # [nframes, bs, latent_dim]
                # mem_pos = self.mem_pos_decoder(z)

                output = self.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                    plot_att_map=plot_att_map,
                    memory_key_padding_mask=~latent_mask if not self.test_efficiency else None, #!! Comment for not FIX
                    # query_pos=query_pos,
                    # pos=mem_pos,
                ).squeeze(0)
                # query_pos = self.query_pos_decoder(queries)
                # # mem_pos = self.mem_pos_decoder(z)
                # output = self.decoder(
                #     tgt=queries,
                #     memory=z,
                #     tgt_key_padding_mask=~mask,
                #     query_pos=query_pos,
                #     # pos=mem_pos,
                # ).squeeze(0)
        
        output = self.final_layer(output) # transforms last dimension from latent_dim to 263 (number of pose features)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        # final shape is [bs, nframes, nfeats], with nframes = max(lengths)
        return feats
