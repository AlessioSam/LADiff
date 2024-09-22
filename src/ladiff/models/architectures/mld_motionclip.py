import os
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

from ladiff.models.operator import PositionalEncoding
from ladiff.utils.temos_utils import lengths_to_mask

from deps.motionclip.src.utils.get_model_and_data import get_model_
from deps.motionclip.src.parser.training import parser

import clip
from deps.motionclip.src.models.architectures.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from deps.motionclip.src.models.modeltype.motionclip import MOTIONCLIP


class MldTextEncoder(nn.Module):

    def __init__(
        self,
        modelpath: str,
        finetune: bool = False,
        last_hidden_state: bool = False,
        latent_dim: list = [1, 256],
    ) -> None:

        super().__init__()

        self.latent_dim = latent_dim

        # Load the model
        #parameters = parser()
        
        clip_model, clip_preprocess = clip.load("ViT-B/32", jit=False) # DEVICE!! # Must set jit=False for training
        clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
        
        for domain in ''.split('_'):
            clip_num_layers = 12
            if domain == 'text':
                clip_model.initialize_parameters()
                clip_model.transformer.resblocks = clip_model.transformer.resblocks[:clip_num_layers]
            if domain == 'image':
                clip_model.initialize_parameters()
                clip_model.visual.transformer = clip_model.transformer.resblocks[:clip_num_layers]


        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        encoder = Encoder_TRANSFORMER()
        decoder = Decoder_TRANSFORMER()

        self.text_model = MOTIONCLIP(encoder, decoder, clip_model=clip_model)


        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath)

        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        self.max_length = self.tokenizer.model_max_length
        if "clip" in modelpath:
            self.text_encoded_dim = self.text_model.config.text_config.hidden_size
            if last_hidden_state:
                self.name = "clip_hidden"
            else:
                self.name = "clip"
        elif "bert" in modelpath:
            self.name = "bert"
            self.text_encoded_dim = self.text_model.config.hidden_size
        else:
            raise ValueError(f"Model {modelpath} not supported")

    def forward(self, texts: List[str]):
        # get prompt text embeddings
        if self.name in ["clip", "clip_hidden"]:
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            # split into max length Clip can handle
            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                text_input_ids = text_input_ids[:, :self.tokenizer.
                                                model_max_length]
        elif self.name == "bert":
            text_inputs = self.tokenizer(texts,
                                         return_tensors="pt",
                                         padding=True)

        # use pooled ouuput if latent dim is two-dimensional
        # pooled = 0 if self.latent_dim[0] == 1 else 1 # (bs, seq_len, text_encoded_dim) -> (bs, text_encoded_dim)
        # text encoder forward, clip must use get_text_features
        if self.name == "clip":
            # (batch_Size, text_encoded_dim)
            text_embeddings = self.text_model.get_text_features(
                text_input_ids.to(self.text_model.device))
            # (batch_Size, 1, text_encoded_dim)
            text_embeddings = text_embeddings.unsqueeze(1)
        elif self.name == "clip_hidden":
            # (batch_Size, seq_length , text_encoded_dim)
            text_embeddings = self.text_model.text_model(
                text_input_ids.to(self.text_model.device)).last_hidden_state
        elif self.name == "bert":
            # (batch_Size, seq_length , text_encoded_dim)
            text_embeddings = self.text_model(
                **text_inputs.to(self.text_model.device)).last_hidden_state
        else:
            raise NotImplementedError(f"Model {self.name} not implemented")

        return text_embeddings
