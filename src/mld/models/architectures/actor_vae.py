from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from mld.utils.temos_utils import lengths_to_mask
from mld.models.operator import PositionalEncoding


class ActorVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 is_vae: bool = True,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.is_vae = is_vae
        input_feats = nfeats
        output_feats = nfeats

        self.dvae = ablation.DVAE
        self.percentage_noised = ablation.PERCENTAGE_NOISED
        self.max_it = ablation.MAX_IT
        self.frame_per_latent = ablation.FRAME_PER_LATENT
        self.joint_distro_fix = ablation.JOINT_DISTRO_FIX
        self.LAD = ablation.LAD
        self.test_efficiency = ablation.TEST_EFFICIENCY

        self.encoder = ActorAgnosticEncoder(nfeats=input_feats,
                                            vae=True,
                                            latent_dim=self.latent_dim,
                                            ff_size=ff_size,
                                            num_layers=num_layers,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            activation=activation,
                                            lad=self.LAD, max_it=self.max_it,
                                            **kwargs)

        self.decoder = ActorAgnosticDecoder(nfeats=output_feats,
                                            vae=True,
                                            latent_dim=self.latent_dim,
                                            ff_size=ff_size,
                                            num_layers=num_layers,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            activation=activation,
                                            **kwargs)

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
    
    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        
        if self.dvae:
            features = self.add_noise(features)

        if self.max_it!=0:
            max_iter_elements = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long)

            dist = self.encoder(features, lengths)

            distro_fix = False
            if distro_fix:
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
                return latent, dist, max_iter_elements
            else:
                mu = dist[0:self.max_it, ...] # mu.shape [max_it, bs, 256]
                logvar = dist[self.max_it:, ...]

                # resampling and padding output
                std = logvar.exp().pow(0.5)
                dist = torch.distributions.Normal(mu, std)
                latent = dist.rsample()
                return latent, dist, max_iter_elements

        else:
            dist = self.encoder(features, lengths)

            if self.is_vae:
                latent = sample_from_distribution(dist)
            else:
                latent = dist.unsqueeze(0)

            return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):

        feats = self.decoder(z, lengths)
        return feats


class ActorAgnosticEncoder(nn.Module):

    def __init__(self,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 lad=False, max_it=5,
                 **kwargs) -> None:
        super().__init__()

        input_feats = nfeats
        self.vae = vae
        self.skel_embedding = nn.Linear(input_feats, latent_dim)
        self.lad = lad
        self.max_it = max_it

        # Action agnostic: only one set of params
        if vae:
            if self.lad:
                self.mu_token = nn.Parameter(torch.randn(self.max_it,latent_dim))
                self.logvar_token = nn.Parameter(torch.randn(self.max_it,latent_dim))
            else:
                self.mu_token = nn.Parameter(torch.randn(latent_dim))
                self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def forward(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        if self.vae:
            if self.lad:
                mu_token = torch.tile(self.mu_token, (bs, 1, 1)).permute(1,0,2)
                logvar_token = torch.tile(self.logvar_token, (bs, 1, 1)).permute(1,0,2)
            else:
                mu_token = torch.tile(self.mu_token, (bs, )).reshape(bs, -1)
                logvar_token = torch.tile(self.logvar_token,
                                        (bs, )).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            if self.lad:
                xseq = torch.cat((mu_token, logvar_token, x), 0)
            else:
                xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)
            

            # create a bigger mask, to allow attend to mu and logvar
            if self.lad:
                token_mask = torch.ones((bs, self.max_it*2), dtype=bool, device=x.device)
                aug_mask = torch.cat((token_mask, mask), 1)
            else:
                token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
                aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs, )).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        

        if self.vae:
            if self.lad:
                dist = final[:self.max_it*2]
                return dist
            else:
                mu, logvar = final[0], final[1]
                std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
                dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return final[0]


class ActorAgnosticDecoder(nn.Module):

    def __init__(self,
                 nfeats: int,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 lad=False, max_it=5,
                 **kwargs) -> None:
        super().__init__()

        output_feats = nfeats
        self.latent_dim = latent_dim
        self.nfeats = nfeats
        self.lad = lad
        self.max_it = max_it

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        # latent_dim = z.shape[1]
        bs, nframes = mask.shape
        nfeats = self.nfeats

        # z = z[None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(nframes,
                                   bs,
                                   self.latent_dim,
                                   device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=time_queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats


def sample_from_distribution(
    dist,
    *,
    fact=1.0,
    sample_mean=False,
) -> Tensor:

    if sample_mean:
        return dist.loc.unsqueeze(0)

    # Reparameterization trick
    if fact is None:
        return dist.rsample().unsqueeze(0)

    # Resclale the eps
    eps = dist.rsample() - dist.loc
    z = dist.loc + fact * eps

    # add latent size
    z = z.unsqueeze(0)
    return z
