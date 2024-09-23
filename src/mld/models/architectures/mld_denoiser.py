import torch
import torch.nn as nn
from torch import  nn
from mld.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (SkipTransformerEncoder,
                                                 TransformerDecoder,
                                                 TransformerDecoderLayer,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
from mld.models.architectures.mdiff_transformer import LinearTemporalDiffusionTransformerDecoderLayer

class MldDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: str = "text",
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 nclasses: int = 10,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = ablation.DIFF_PE_TYPE
        self.idea = ablation.IDEA
        self.MD_trans = ablation.MD_TRANS

        self.test_efficiency = ablation.TEST_EFFICIENCY

        
        if self.diffusion_only:
            # assert self.arch == "trans_enc", "only implement encoder for diffusion-only"
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, nfeats)

        # emb proj
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time from text_encoded_dim to latent_dim
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                    self.latent_dim)
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Sequential(
                    nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
        elif self.condition in ['action']:
            self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(self.latent_dim,
                                                    self.latent_dim)
            self.emb_proj = EmbedAction(nclasses,
                                        self.latent_dim,
                                        guidance_scale=guidance_scale,
                                        guidance_uncodp=guidance_uncondp)
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        


        if self.pe_type == "actor":
            self.query_pos = PositionalEncoding(self.latent_dim, dropout)
            self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        if self.arch == "trans_enc": # this
            if self.ablation_skip_connection:
                # use DETR transformer
                if self.MD_trans:
                    encoder_layer = LinearTemporalDiffusionTransformerDecoderLayer(
                        d_model=self.latent_dim,
                        text_latent_dim=self.latent_dim,
                        time_embed_dim=self.latent_dim,
                        ffn_dim=ff_size,
                        num_head=num_heads,
                        dropout=dropout
                    )
                
                else:
                    encoder_layer = TransformerEncoderLayer(
                        self.latent_dim,
                        num_heads,
                        ff_size,
                        dropout,
                        activation,
                        normalize_before
                    )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)
        elif self.arch == "trans_dec":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")

    def forward(self,
                sample, # 
                timestep,
                encoder_hidden_states,
                enclat=None, 
                enclat_future=None,
                lengths=None, latent_idx=None,
                max_iter_elements=None,
                remo_cond=None,
                remo_text_cond=None,
                **kwargs):

        # compute dist_masks for attention
        if not self.test_efficiency:
            if max_iter_elements is not None:
                max_iter = sample.shape[1]
                latent_mask = torch.ones((len(max_iter_elements), max_iter), #! was ones
                                        dtype=bool,
                                        device=sample.device)
                for i, max_iter_element in enumerate(max_iter_elements):
                    latent_mask[i, max_iter_element:] = False #! was False

        # 0.  dimension matching
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        text_remo_emb = None #!
        

        # 0. check lengths for no vae (diffusion only), not used by LAD
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        if self.condition in ["text", "text_uncond"]:
            # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            if remo_text_cond is not None:
                remo_text_cond = remo_text_cond.permute(1, 0, 2)
                if remo_text_cond.shape[1] != encoder_hidden_states.shape[1]:
                    remo_text_cond = remo_text_cond.reshape(-1, encoder_hidden_states.shape[1], encoder_hidden_states.shape[2])
                text_remo_emb = self.emb_proj(remo_text_cond)

            text_emb = encoder_hidden_states  # [1, bs, latent_dim]
            # textembedding projection
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_emb)
            else:
                text_emb_latent = text_emb
            if self.abl_plus:
                emb_latent = time_emb + text_emb_latent
            else:
                emb_latent = torch.cat((time_emb, text_emb_latent), 0) # not used by MD_trans

        elif self.condition in ['action']:
            action_emb = self.emb_proj(encoder_hidden_states)
            if self.abl_plus:
                emb_latent = action_emb + time_emb
            else:
                emb_latent = torch.cat((time_emb, action_emb), 0)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # 4. transformer
                #! IDEA

        
        if self.idea=='ard' and enclat is not None:
            enclat = enclat.permute(1, 0, 2) 
        #if enclat_future is not None:
        #    enclat_future = enclat_future.permute(1, 0, 2) 

        if self.arch == "trans_enc": #this
            if self.diffusion_only:
                sample = self.pose_embd(sample)
                xseq = torch.cat((emb_latent, sample), axis=0)
            else:
                if not self.MD_trans:
                    if self.idea=='ard' and enclat is not None:
                        xseq = torch.cat((sample, enclat, emb_latent), axis=0)
                    else:
                        xseq = torch.cat((sample, emb_latent), axis=0)

                    #if enclat_future is not None:
                    #    xseq = torch.cat((enclat_future, xseq), axis=0)   
                

            # if self.ablation_skip_connection:
            #     xseq = self.query_pos(xseq)
            #     tokens = self.encoder(xseq)
            # else:
            #     # adding the timestep embed
            #     # [seqlen+1, bs, d]
            #     # todo change to query_pos_decoder
            
            if self.MD_trans:
                if enclat is not None:
                    xseq = torch.cat((sample, enclat), axis=0)
                else:
                    xseq = sample
                xseq = self.query_pos(xseq)
                if remo_cond!=None:
                    #! text_emb_latent = torch.cat((text_emb_latent, remo_cond), axis=0)
                    if text_remo_emb is not None:
                        text_emb_latent = torch.cat((text_emb_latent, text_remo_emb, remo_cond), axis=0)
                    else:
                        if remo_cond.shape[1] != text_emb_latent.shape[1]:
                            to_replicate = int(text_emb_latent.shape[1]/remo_cond.shape[1])
                            remo_cond = remo_cond.repeat(1, to_replicate, 1)
                            remo_cond = remo_cond.reshape(-1, text_emb_latent.shape[1], text_emb_latent.shape[2])

                        text_emb_latent = torch.cat((text_emb_latent, remo_cond), axis=0)
                    # Create mask
                    mask_cond = latent_mask.clone()

                    #! Added from here for eccv, if not added then not works for multiple retrieval or text retrieval
                    if mask_cond.shape[1] != remo_cond.shape[0]:
                        to_add = int(remo_cond.shape[0]/mask_cond.shape[1])
                        for i in range(to_add - 1):
                            mask_cond = torch.cat((mask_cond, latent_mask), axis=1)

                    if text_remo_emb is not None:
                        if mask_cond.shape[1] != text_emb_latent.shape[0]:
                            to_add = int(text_emb_latent.shape[0]-mask_cond.shape[1])-1
                            extra_mask = torch.ones((mask_cond.shape[0], to_add), dtype=bool, device=mask_cond.device)
                            mask_cond = torch.cat((extra_mask,mask_cond), axis=1)
                    #else:
                    #    to_add = 1
                    #    extra_mask = torch.ones((mask_cond.shape[0], to_add), dtype=bool, device=mask_cond.device)
                    #    mask_cond = torch.cat((extra_mask,mask_cond), axis=1)
                    
                tokens = self.encoder(src=xseq, xf=text_emb_latent, 
                                      emb=time_emb, MD_trans=self.MD_trans,
                                      src_key_padding_mask=~latent_mask 
                                        if (max_iter_elements is not None and not self.test_efficiency) else None,
                                        tgt_key_padding_mask=~mask_cond if remo_cond!=None else None)
                
                
            else:
                xseq = self.query_pos(xseq)
                tokens = self.encoder(xseq)


            if self.diffusion_only:
                sample = tokens[emb_latent.shape[0]:]
                sample = self.pose_proj(sample)

                # zero for padded area
                sample[~mask.T] = 0
            else:
                if enclat_future is not None:
                    sample = tokens[latent_idx].unsqueeze(0)
                else:
                    sample = tokens[:sample.shape[0]] #takes all

        elif self.arch == "trans_dec":
            if self.diffusion_only:
                sample = self.pose_embd(sample)

            # tgt    - [1 or 5 or 10, bs, latent_dim]
            # memory - [token_num, bs, latent_dim]
            sample = self.query_pos(sample)
            emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)

            if self.diffusion_only:
                sample = self.pose_proj(sample)
                # zero for padded area
                sample[~mask.T] = 0
        else:
            raise TypeError("{self.arch} is not supoorted")

        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        sample = sample.permute(1, 0, 2)
        

        return (sample, )


class EmbedAction(nn.Module):

    def __init__(self,
                 num_actions,
                 latent_dim,
                 guidance_scale=7.5,
                 guidance_uncodp=0.1,
                 force_mask=False):
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        if not self.training and self.guidance_scale > 1.0:
            uncond, output = output.chunk(2)
            uncond_out = self.mask_cond(uncond, force=True)
            out = self.mask_cond(output)
            output = torch.cat((uncond_out, out))

        output = self.mask_cond(output)

        return output.unsqueeze(0)

    def mask_cond(self, output, force=False):
        bs, d = output.shape
        # classifer guidence
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) *
                self.guidance_uncodp).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return output * (1. - mask)
        else:
            return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
