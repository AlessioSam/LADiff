import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding
from mld.models.operator.position_encoding import PositionEmbeddingSine, PositionEmbeddingSine1D
import torch.nn.functional as F
import clip

from .base import BaseModel

class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.save_latents = cfg.TEST.SAVE_LATENTS
        self.datamodule = datamodule

        self.motion_conditioning = cfg.model.motion_conditioning
        self.vae_from_t2m = cfg.model.vae_from_t2m
        self.sample_latent_idx = cfg.model.sample_latent_idx
        self.force_seq = cfg.FORCE_SEQ
        self.subphase = cfg.TRAIN.SUBPHASE
        self.nframes = cfg.TRAIN.N_FRAMES

        self.test_efficiency = cfg.TRAIN.ABLATION.TEST_EFFICIENCY

        self.subphase = None if self.subphase == 'None' else self.subphase
        self.nframes = None if self.nframes == 'None' else self.nframes
        
        self.max_it = cfg.TRAIN.ABLATION.MAX_IT
        self.frame_per_latent = cfg.TRAIN.ABLATION.FRAME_PER_LATENT

        self.joint_distro_fix = cfg.TRAIN.ABLATION.JOINT_DISTRO_FIX

        self.ARDIFF = cfg.ARDIFF
        self.LAD = cfg.TRAIN.ABLATION.LAD

        self.Loss = torch.nn.SmoothL1Loss()

        #self.latent_dim_stage1 = cfg.model.latent_dim_stage1 if self.subphase==None else cfg.TRAIN.N_FRAMES
        self.latent_dim_stage1 = cfg.model.latent_dim_stage1

        self.pe_latent = cfg.TRAIN.ABLATION.PE_LATENT
        if self.pe_latent:
            self.pos_encoding_latent = PositionEmbeddingSine1D(d_model=self.latent_dim[-1], max_len=self.max_it, batch_first=True)

        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        if self.vae_type != "no":
            self.vae = instantiate_from_config(cfg.model.motion_vae)

        

        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["mld", "vposert","actor"]:
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler) 

        if self.condition in ["text", "text_uncond"]:
            self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond']:
            self.feats2joints = datamodule.feats2joints
        elif self.condition == 'action':
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.DATASET.SMPL_PATH)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='vertices',
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
        
        self.remo = cfg.TRAIN.ABLATION.REMO
        if self.remo:
            self.DATASET_ = cfg.TRAIN.DATASETS
            print("Loading retrieval database: ", self.DATASET_)
            if 'humanml3d' in self.DATASET_:
                data_remo = np.load('./ReMoDiffuse/data/database/t2m_text_train.npz')
            else:
                data_remo = np.load('./ReMoDiffuse/data/database/kit_text_train.npz')
            
            self.text_features = torch.Tensor(data_remo['text_features'])
            self.captions = data_remo['captions']
            self.motions = data_remo['motions']
            self.m_lengths = data_remo['m_lengths']
            self.clip_seq_features = data_remo['clip_seq_features']
            self.train_indexes = data_remo.get('train_indexes', None)
            self.test_indexes = data_remo.get('test_indexes', None)
            print("Loaded retrieval database")

            print("Loading clip remo")
            import clip
            from torch import layer_norm, nn
            self.clip_remo, _ = clip.load('ViT-B/32', "cpu")
            for param in self.clip_remo.parameters():
                param.requires_grad = False
            # Create text encoder for remodiffuse
            self.use_text_proj = False
            self.num_retrieval = 1 #! PUT 1 FOR REBUTTAL
            self.results = {}
            self.text_pre_proj = nn.Linear(512, 256)
            
            textTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=2048,
                dropout=0,
                activation='gelu')
            self.textTransEncoder = nn.TransformerEncoder(
                textTransEncoderLayer,
                num_layers=2)
            self.text_ln = nn.LayerNorm(256)

            # Load weights from remodiffuse and put self.text_pre_proj weights
            if 'humanml3d' in self.DATASET_:
                remodiffuse_weights = torch.load('./ReMoDiffuse/logs/remodiffuse_t2m/latest.pth')
            else:
                remodiffuse_weights = torch.load('./ReMoDiffuse/logs/remodiffuse_kit/latest.pth')
            remodiffuse_weights = remodiffuse_weights['state_dict']
            
            for key in self.textTransEncoder.state_dict().keys():
                self.textTransEncoder.state_dict()[key] = remodiffuse_weights['model.textTransEncoder.'+key]
            
            for key in self.text_pre_proj.state_dict().keys():
                self.text_pre_proj.state_dict()[key] = remodiffuse_weights['model.text_pre_proj.'+key]

            for key in self.text_ln.state_dict().keys():
                self.text_ln.state_dict()[key] = remodiffuse_weights['model.text_ln.'+key]
            
            for param in self.textTransEncoder.parameters():
                param.requires_grad = False

            for param in self.text_pre_proj.parameters():
                param.requires_grad = False
            
            for param in self.text_ln.parameters():
                param.requires_grad = False

            print("Loaded clip remo")


    def extract_text_feature(self, text, clip_model, device):
            text = clip.tokenize([text], truncate=True).to(device)
            with torch.no_grad():
                #!text_features = clip_model.encode_text(text)
                #text_features = clip_model(text)
                text_features = self.clip_remo.encode_text(text)
            return text_features    
     
    def encode_text(self, text, clip_feat=None, device=None):
        B = len(text)
        text = clip.tokenize(text, truncate=True).to(device)
        if clip_feat is None:
            with torch.no_grad():
                x = self.clip_remo.token_embedding(text).type(self.clip_remo.dtype)  # [batch_size, n_ctx, d_model]

                x = x + self.clip_remo.positional_embedding.type(self.clip_remo.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip_remo.transformer(x)
                x = self.clip_remo.ln_final(x).type(self.clip_remo.dtype)
                return x#.permute(1, 0, 2)
        else:
            x = clip_feat.type(self.clip_remo.dtype).to(device).permute(1, 0, 2)

        # T, B, D
        '''
        with torch.no_grad(): #! messo io
            x = self.text_pre_proj(x)
            xf_out = self.textTransEncoder(x)
            xf_out = self.text_ln(xf_out)
            if self.use_text_proj:
                xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
                # B, T, D
                xf_out = xf_out.permute(1, 0, 2)
                return xf_proj, xf_out
            else:
                xf_out = xf_out.permute(1, 0, 2)
                return xf_out'''

    def retrieve(self, caption, length, clip_model=None, device=None, idx=None):
        value = hash(caption)
        if value in self.results:
            return self.results[value]
        text_feature = self.extract_text_feature(caption, clip_model, device)
        
        rel_length = torch.LongTensor(self.m_lengths).to(device)
        rel_length = torch.abs(rel_length - length) / torch.clamp(rel_length, min=length)
        
        semantic_score = F.cosine_similarity(self.text_features.to(device), text_feature)
        kinematic_score = torch.exp(-rel_length * 0.1)
        score = semantic_score * kinematic_score
        indexes = torch.argsort(score, descending=True)
        data = []
        cnt = 0
        for idx in indexes:
            caption, motion, m_length = self.captions[idx], self.motions[idx], self.m_lengths[idx]
            #!if not self.training or m_length != length:
            if m_length != length: #! era !=
                cnt += 1
                data.append(idx.item())
                if cnt == self.num_retrieval:
                    self.results[value] = data
                    return data 

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname,
                         "text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

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

    def forward(self, batch, latentwise_gen=None, plot_att_map=None):
        texts = batch["text"]
        lengths = batch["length"]
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m, max_iter_elements = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                if latentwise_gen:
                    lengths = lengths * self.max_it
                    z = z.repeat(1, self.max_it, 1) # repeat once for each row of a latent
                    if latentwise_gen == "fw":
                        for idx in range(self.max_it): # progressively pad latent rows
                            z[idx+1:, idx, :] = 0
                    elif latentwise_gen == "bw":
                        for idx in range(self.max_it): # regressively pad latent rows
                            z[:self.max_it-(idx+1), idx, :] = 0
                feats_rst = self.vae.decode(z, lengths, plot_att_map=plot_att_map, latentwise_gen=latentwise_gen)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        joints = self.feats2joints(feats_rst.detach().cpu()) # from [batch_size, n_frames, 263] to [batch_size, n_frames, 22, 3]
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist, max_iter_elements = self.vae.encode(feats_ref, length)
        feats_rst, max_iter_elements = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, 
                            remo_cond=None, remo_text_cond=None):
        # reverse
        if self.cfg.IDEA == 'ard':

            bsz = encoder_hidden_states.shape[0]
           
            if self.do_classifier_free_guidance:
                bsz = bsz // 2
            
            #if self.subphase == 'stage1' or self.subphase=='stage2':
            if self.nframes!=None:
                max_lng = max(lengths)
                resto = max_lng % self.latent_dim_stage1
                if resto != 0:
                    ar_iterations = max_lng // self.latent_dim_stage1 + 1
                else:
                    ar_iterations = max_lng // self.latent_dim_stage1
            else:
                max_lng = max(lengths)
                resto = max_lng % self.frame_per_latent
                if resto != 0:
                    ar_iterations = max_lng // self.frame_per_latent + 1
                else:
                    ar_iterations = max_lng // self.frame_per_latent
            #else:
            #    ar_iterations = self.latent_dim[0]

            if self.ARDIFF:
                latents = torch.randn(
                        (bsz, ar_iterations, self.latent_dim[-1]),
                        device=encoder_hidden_states.device,
                        dtype=torch.float,
                    )
            else:
                
                if self.joint_distro_fix:
                    latents = torch.tensor([], device=encoder_hidden_states.device)
                    max_iter_elements = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long) # (torch.tensor(lengths) // self.frame_per_latent + 1)
                    for i, max_iter_elem in enumerate(max_iter_elements):
                        latent = torch.randn(1, max_iter_elem, self.latent_dim[-1], device=encoder_hidden_states.device)
                        # Fill the rest with zeros
                        zeros = torch.zeros(1, self.max_it - max_iter_elem, self.latent_dim[-1], device=encoder_hidden_states.device)
                        latent = torch.cat((latent, zeros), dim=1)
                        latents = torch.cat((latents, latent), dim=0) if i > 0 else latent

                else:
                    max_iter_elements = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long) # (torch.tensor(lengths) // self.frame_per_latent + 1)
                    latents = torch.randn(
                        (bsz, self.max_it if not self.test_efficiency else max_iter_elements[0].item()
                         , self.latent_dim[-1]),
                        device=encoder_hidden_states.device,
                        dtype=torch.float,
                    )
                    if not self.test_efficiency:
                        if self.LAD:
                            # max_iter_elements = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long) # (torch.tensor(lengths) // self.frame_per_latent + 1)
                            for i, max_iter_elem in enumerate(max_iter_elements):
                                latents[i, max_iter_elem:] = 0

                # 
                #!! BEFORE FIX, COMMENT ABOVE AND UNCOMMENT BELOW
                #latents = torch.randn(
                #        (bsz, self.max_it, self.latent_dim[-1]),
                #        device=encoder_hidden_states.device,
                #        dtype=torch.float,
                #    )
                # for i, max_iter_elem in enumerate(max_iter_elements):
                    #latents[i, max_iter_elem:] = 0

                

            

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

            # set timesteps
            self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
            timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, and between [0, 1]
            extra_step_kwargs = {}
            if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
            
            if self.ARDIFF:
                final_latents = torch.tensor([], device=encoder_hidden_states.device)
                
                for time_batch in range(ar_iterations):
                    latents_batch = latents[:, time_batch, :].unsqueeze(1)

                    if time_batch > 0:
                        if self.motion_conditioning == 'full' or self.motion_conditioning == 'middle':
                            enclat = final_latents[:, :time_batch, :] #.permute(1, 0, 2)
                        elif self.motion_conditioning == 'last':
                            enclat = final_latents[:, time_batch-1:time_batch, :]
    
                        enclat = (torch.cat([enclat] * 2) if self.do_classifier_free_guidance else enclat)

                    else:
                        enclat = None
                        #enclat_future = None


                    for i, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = (torch.cat(
                            [latents_batch] *
                            2) if self.do_classifier_free_guidance else latents_batch)
                        lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                                        else lengths)
                        # predict the noise residual

                        noise_pred = self.denoiser(
                            sample=latent_model_input,
                            timestep=t,
                            encoder_hidden_states=encoder_hidden_states,
                            enclat=enclat,
                            #enclat_future= enclat_future,
                            lengths=lengths_reverse, 
                            remo_cond=remo_cond,
                            remo_text_cond=remo_text_cond,
                        )[0]
                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (
                                noise_pred_text - noise_pred_uncond)
                        latents_batch = self.scheduler.step(noise_pred, t, latents_batch,
                                                        **extra_step_kwargs).prev_sample

                    final_latents = torch.cat((final_latents, latents_batch), dim=1) if time_batch > 0 else latents_batch
                
                # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
                # latents = final_latents.permute(0, 2, 1) # T2M
                latents = final_latents.permute(1, 0, 2) # MLD   

            else:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (torch.cat(
                        [latents] *
                        2) if self.do_classifier_free_guidance else latents)
                    lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                                    else lengths)
                    # predict the noise residual

                    noise_pred = self.denoiser(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=encoder_hidden_states,
                        enclat=None,
                        lengths=lengths_reverse,
                        max_iter_elements=torch.cat([max_iter_elements]*2) if self.do_classifier_free_guidance else max_iter_elements, #comment or set to None to not use masks in denoiser
                        remo_cond=torch.cat([remo_cond.permute(1,0,2)]*2).permute(1,0,2) if
                                (remo_cond is not None and self.do_classifier_free_guidance) else remo_cond,
                    )[0]
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond)
                    latents = self.scheduler.step(noise_pred, t, latents,
                                                    **extra_step_kwargs).prev_sample
                    # KEEP SETTING EXTRA LATENTS TO ZERO
                    # max_iter_elements = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long) # (torch.tensor(lengths) // self.frame_per_latent + 1)
                    # for i, max_iter_elem in enumerate(max_iter_elements):
                    #     latents[i, max_iter_elem:] = 0
                    # END KEEP SETTING EXTRA LATENTS TO ZERO

                # [batch_size, max_it, latent_dim] -> [max_it, batch_size, latent_dim]
                latents = latents.permute(1, 0, 2)
               
                

             
        else:
            bsz = encoder_hidden_states.shape[0]
            if self.do_classifier_free_guidance:
                bsz = bsz // 2
            if self.vae_type == "no":
                assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
                latents = torch.randn(
                    (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                    device=encoder_hidden_states.device,
                    dtype=torch.float,
                )
            else:
                latents = torch.randn(
                    (bsz, self.latent_dim[0], self.latent_dim[-1]),
                    device=encoder_hidden_states.device,
                    dtype=torch.float,
                )

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            # set timesteps
            self.scheduler.set_timesteps(
                self.cfg.model.scheduler.num_inference_timesteps)
            timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, and between [0, 1]
            extra_step_kwargs = {}
            if "eta" in set(
                    inspect.signature(self.scheduler.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (torch.cat(
                    [latents] *
                    2) if self.do_classifier_free_guidance else latents)
                lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                                else lengths)
                # predict the noise residual
                noise_pred = self.denoiser(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    enclat=None,
                    lengths=lengths_reverse,
                )[0]
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents,
                                                **extra_step_kwargs).prev_sample

            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents = latents.permute(1, 0, 2)

        # USELESS IF WE KEEP SETTING EXTRA LATENTS TO ZERO ABOVE (AlessioP) -> we are forcing again to zero (AlessioS)
        if self.max_it!=0:
            if self.LAD:
                max_it_el = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long) # (torch.tensor(lengths) // self.frame_per_latent + 1)
                for i, max_iter_elem in enumerate(max_it_el):
                    latents[max_iter_elem:, i] = 0
                if self.ARDIFF:
                    if latents.shape[0]<self.max_it:
                        latents = torch.cat((latents, torch.zeros((self.max_it-latents.shape[0], latents.shape[1], latents.shape[2]), device=latents.device)), dim=0)
        
        return latents
    
    def _diffusion_reverse_tsne(self, encoder_hidden_states, lengths=None):
        # encoder_hidden_states: [batch_size, seq_len, hidden_size]
        if self.cfg.IDEA == 'ard':
            bsz = encoder_hidden_states.shape[0]
            
            if self.do_classifier_free_guidance:
                bsz = bsz // 2
            
            #if self.subphase == 'stage1' or self.subphase=='stage2':
            if self.nframes!=None:
                max_lng = max(lengths)
                resto = max_lng % self.latent_dim_stage1
                if resto != 0:
                    ar_iterations = max_lng // self.latent_dim_stage1 + 1
                else:
                    ar_iterations = max_lng // self.latent_dim_stage1
            else:
                max_lng = max(lengths)
                resto = max_lng % self.frame_per_latent
                if resto != 0:
                    ar_iterations = max_lng // self.frame_per_latent + 1
                else:
                    ar_iterations = max_lng // self.frame_per_latent
            #else:
            #    ar_iterations = self.latent_dim[0]

            latents = torch.randn(
                        (bsz, ar_iterations, self.latent_dim[-1]),
                        device=encoder_hidden_states.device,
                        dtype=torch.float,
                    )

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

            # set timesteps
            self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
            timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, and between [0, 1]
            extra_step_kwargs = {}
            if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
            

            final_latents = torch.tensor([], device=encoder_hidden_states.device)

            # reverse
            latents_t = []

            for time_batch in range(ar_iterations):
                latents_batch = latents[:, time_batch, :].unsqueeze(1)

                if time_batch > 0:
                    if self.motion_conditioning == 'full' or self.motion_conditioning == 'middle':
                        enclat = final_latents[:, :time_batch, :] #.permute(1, 0, 2)
                    elif self.motion_conditioning == 'last':
                        enclat = final_latents[:, time_batch-1:time_batch, :]
  
                    enclat = (torch.cat([enclat] * 2) if self.do_classifier_free_guidance else enclat)

                else:
                    enclat = None
                    #enclat_future = None

                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (torch.cat(
                        [latents_batch] *
                        2) if self.do_classifier_free_guidance else latents_batch)
                    lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                                    else lengths)
                    # predict the noise residual
                    # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.denoiser(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=encoder_hidden_states,
                        enclat=enclat,
                        #enclat_future= enclat_future,
                        lengths=lengths_reverse,
                    )[0]

   
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond)
                    latents_batch = self.scheduler.step(noise_pred, t, latents_batch,
                                                    **extra_step_kwargs).prev_sample
                    
                    latents_t.append(latents_batch.permute(1,0,2))
                    final_latents = torch.cat((final_latents, latents_batch), dim=1) if time_batch > 0 else latents_batch
                
            latents_t = torch.cat(latents_t)

            
        else:
            # init latents
            bsz = encoder_hidden_states.shape[0]
            if self.do_classifier_free_guidance:
                bsz = bsz // 2
            if self.vae_type == "no":
                assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
                latents = torch.randn(
                    (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                    device=encoder_hidden_states.device,
                    dtype=torch.float,
                )
            else:
                latents = torch.randn(
                    (bsz, self.latent_dim[0], self.latent_dim[-1]),
                    device=encoder_hidden_states.device,
                    dtype=torch.float,
                )

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            # set timesteps
            self.scheduler.set_timesteps(
                self.cfg.model.scheduler.num_inference_timesteps)
            timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, and between [0, 1]
            extra_step_kwargs = {}
            if "eta" in set(
                    inspect.signature(self.scheduler.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

            # reverse
            latents_t = []
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (torch.cat(
                    [latents] *
                    2) if self.do_classifier_free_guidance else latents)
                lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                                else lengths)
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                noise_pred = self.denoiser(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    lengths=lengths_reverse,
                )[0]
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                    # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                    #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
                latents = self.scheduler.step(noise_pred, t, latents,
                                                **extra_step_kwargs).prev_sample
                # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
                latents_t.append(latents.permute(1,0,2))
            # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
            latents_t = torch.cat(latents_t)

        if self.max_it!=0:
            max_it_el = torch.ceil(torch.tensor(lengths) / self.frame_per_latent).to(torch.long) # (torch.tensor(lengths) // self.frame_per_latent + 1)
            for i, max_iter_elem in enumerate(max_it_el):
                latents[max_iter_elem:, i] = 0
            if latents.shape[0]<self.max_it:
                latents = torch.cat((latents, torch.zeros((self.max_it-latents.shape[0], latents.shape[1], latents.shape[2]), device=latents.device)), dim=0)

        return latents_t

    def _diffusion_process(self, latents, encoder_hidden_states,
                            lengths=None, cond_z=None, cond_z_future=None,
                            latent_idx=None, max_iter_elements=None,
                            remo_cond=None, remo_text_cond=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        
        # latent now : [batch_size, 8, 512]
        #!! latents = latents.permute(1, 0, 2) # UNCOMMENT FOR ORIGINAL CODE

            
        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
       
        
        
       
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        
        if self.ARDIFF==False:
            if self.LAD:
                for i, max_iter_element in enumerate(max_iter_elements):
                    noisy_latents[i, max_iter_element:] = 0
                    #if remo_cond!=None:
                    #    remo_cond[i, max_iter_element:] = 0

        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            enclat=cond_z,
            enclat_future=cond_z_future,
            lengths=lengths, latent_idx=latent_idx,
            return_dict=False,
            max_iter_elements=max_iter_elements, #comment or set to None to not use masks in denoiser
            remo_cond=remo_cond,
            remo_text_cond=remo_text_cond,
        )[0] 


        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents

        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

      

        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m, max_iter_elements = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
          
            
            
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm, max_iter_elements = self.vae.encode(feats_rst, lengths)

        # joints recover
        if self.condition == "text":
            joints_rst = self.feats2joints(feats_rst)
            joints_ref = self.feats2joints(feats_ref)
        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        dist_ref = None
        if dist_m is not None:
            if self.is_vae:
                if self.joint_distro_fix:
                    dist_ref_list = []
                    for i, max_iter_element in enumerate(max_iter_elements):
                        mu_ref = torch.zeros_like(dist_m[i].loc[:max_iter_element, :])
                        scale_ref = torch.ones_like(dist_m[i].scale[:max_iter_element, :])
                        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
                        dist_ref_list.append(dist_ref)
                    dist_ref = dist_ref_list
                else:
                    # Create a centred normal distribution to compare with
                    mu_ref = torch.zeros_like(dist_m.loc)
                    scale_ref = torch.ones_like(dist_m.scale)
                    dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set
    

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"] # shape is [batch_size, n_frames, 263 (feats)]
        lengths = batch["length"]

        is_starting = batch["is_starting"]
       
      
        
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                if self.subphase == None:
                    z, dist, max_iter_elements = self.vae.encode(feats_ref, lengths)
                elif self.subphase == 'stage2' or self.subphase == 'stage2_1':
                    z_final = torch.tensor([], device=feats_ref.device)
                    # Pass through the motion encoder self.nframes at time
                    for i in range(0, feats_ref.shape[1], self.nframes):
                        if i + self.nframes > feats_ref.shape[1]:
                            break
                        new_lenghts = torch.tensor([self.nframes] * feats_ref.shape[0], device=feats_ref.device)
                        z, dist_m, max_iter_elements = self.vae.encode(feats_ref[:, i:i+self.nframes], new_lenghts)
                        z_final = torch.cat((z_final, z), dim=0) if z_final.shape[0] != 0 else z # [num of latents, bs, latent features]
                    
                    # Calcola resto
                    rest = feats_ref.shape[1] % self.nframes
                    if rest != 0:
                        new_lenghts = torch.tensor([rest] * feats_ref.shape[0], device=feats_ref.device)
                        z, dist_m, max_iter_elements = self.vae.encode(feats_ref[:, -rest:], new_lenghts)
                        z_final = torch.cat((z_final, z), dim=0) if z_final.shape[0] != 0 else z
                    
                    z = z_final
                   

                if self.vae_from_t2m:
                    print('ADJUST DIMENSIONALITY!')
                    quit()
   
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
                
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]

            if self.remo:
                # RETRIEVES THE MOTION
                B = len(text)
                all_indexes = []
                for b_ix in range(B): # 3.1sec per bs=32
                    length = int(lengths[b_ix])
                    batch_indexes = self.retrieve(text[b_ix], length, clip_model=None, device = z.device)
                    all_indexes.extend(batch_indexes)
                all_indexes = np.array(all_indexes)
                all_motions = torch.Tensor(self.motions[all_indexes]).to(z.device) # [batch, n_frames, 263]
                all_m_lengths = torch.Tensor(self.m_lengths[all_indexes]).long() # [batch]
                #all_captions = self.captions[all_indexes].tolist() # [batch, 5] #! REMOVE FOR REBUTTAL
                
                # Encode retrieved motion with VAE
                z_remo, _, _ = self.vae.encode(all_motions[:,:torch.max(all_m_lengths)], all_m_lengths) # [5, B*num_retrieval, 256]
                #if z_remo.shape[1] != B:
                #    z_remo = z_remo.reshape(-1, B, z_remo.shape[2])

                # Encode retrieved captions with text_encoder
                #z_remo_text = self.text_encoder(all_captions) # [B*num_retrieval, 1, 768] #! REMOVE FOR REBUTTAL

            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)

        elif self.condition in ['action']:
            action = batch['action']
            # text encode
            cond_emb = action
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion process return with noise and noise_pred
        #!! CHANGE THIS IN ORDER TO BE AUTOREGRESSIVE
        if self.cfg.IDEA == 'ard':

            # T2M: z.shape = [batch_size, n_token=512, latent_dim=8] -> z = z.permute(0,2,1) # [batch_size, latent_dim=8, n_token=512]
            # MLD: z.shape = [latent_dim, batch_size, n_token] -> z = z.permute(1,0,2) # [batch_size, latent_dim, n_token]
            z = z.permute(1,0,2) # [batch_size, latent_dim, n_token]

            latent_dimensionality = z.shape[1] 
           

            if self.nframes==None:
                # Chose a value between 1 and max_iter_elements
                if self.ARDIFF:
                    latent_idx = []
                    for i, ind in enumerate(max_iter_elements):
                        latent_idx.append(torch.randint(1,ind,(1, )).item())

                    # Define a temporal encoding that helps the diffusion process to understand the order of the latent
                    if self.pe_latent:
                        pos_encoding_latent = self.pos_encoding_latent(z.permute(1,0,2))
                        z = z.clone() + pos_encoding_latent


                    # CREATE CONDITIONING
                    cond_z = torch.zeros(z.shape[0], 1, z.shape[2], device=z.device) # [batch_size, 1, 256]
                    z_new = torch.zeros(z.shape[0], 1, z.shape[2], device=z.device) # [batch_size, 1, 256]
                    for i, l_idx in enumerate(latent_idx):
                        cond_z[i] = z[i,l_idx-1:l_idx]
                        z_new[i] = z[i,l_idx]
                    cond_z_future = None

                    coin = np.random.uniform(0, 1)
                    if coin < 0.33: #(1/max(max_iter_elements)): # 0.33: # For rigth coin it depends on batch
                        z_new = z[:,0:1]
                        cond_z = None

                    z = z_new
                else:
                    cond_z = None
                    cond_z_future = None
                    
            else:
                # Sample a number of latent to condition on
                if self.sample_latent_idx == 'single':
                    if self.subphase==None or self.subphase=='stage2_1':
                        latent_idx = torch.randint(0,latent_dimensionality,(1, ),device=z.device)
                        
                    elif self.subphase=='stage2':
                        
                        #latent_idx = torch.tensor([]).to(z.device)
                        #for ist in range(len(is_starting)):
                        #    if is_starting[ist] == True:
                        #        latent_idx = torch.cat((latent_idx, torch.tensor([0]).to(z.device)), dim=0)
                        #    else:
                        #        latent_idx = torch.cat((latent_idx, torch.tensor([1]).to(z.device)), dim=0)

                        # Latent_idx is the last latent dimension
                        latent_idx = torch.tensor([1], device=z.device)
                        
                        
                elif self.sample_latent_idx == 'multi':
                    print('SAMPLE MULTI NOT IMPLEMENTED YET')
                    quit()
                    # Sample a number for each motion between 0 and the number of latent dimensions
                    latent_idx = torch.randint(0,latent_dimensionality,(z.shape[0], ),device=z.device)

                # CREATE CONDITIONING 
                if latent_idx.item() > 0:
                    if self.motion_conditioning == 'full' or self.motion_conditioning == 'middle':
                        cond_z = z[:,:latent_idx]
                    elif self.motion_conditioning == 'last':
                        cond_z = z[:,latent_idx-1:latent_idx]
                else:
                    cond_z = None

                if self.motion_conditioning == 'middle':
                    cond_z_future = z[:, latent_idx+1:, :]      
                else:
                    cond_z_future = None


                #coin = np.random.uniform(0, 1)
                #if coin < .33: # 0.33:
                #    latent_idx = torch.tensor([0], device=z.device)
                #    cond_z = None

             
                # LATENT TO DIFFUSE
                z = z[:,latent_idx]  # [batch_size, 1, n_token]
          


            n_set = self._diffusion_process(z, cond_emb, lengths=None, 
                                            cond_z=cond_z, cond_z_future=cond_z_future, 
                                            max_iter_elements=max_iter_elements, 
                                            remo_cond=z_remo if self.remo else None,)
                                            #remo_text_cond=z_remo_text if self.remo else None) #! REMOVE FOR REBUTTAL
                                            
        else:
            n_set = self._diffusion_process(z, cond_emb, lengths)

        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                                     dtype=batch['action'].dtype))
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)
                

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                if self.subphase!='stage2' or self.subphase!='stage1':
                    feats_rst = self.vae.decode(z, lengths)
                else:
                    feats_rst = torch.tensor([], device=z.device)
                    # Pass through the motion decoder
                    for i in range(0, z.shape[0]):
                        new_lenghts = torch.tensor([self.nframes] * z.shape[1], device=z.device)
                        feats = self.vae.decode(z[i].unsqueeze(0), new_lenghts)
                        feats_rst = torch.cat((feats_rst, feats), dim=1) if feats_rst.shape[0] != 0 else feats

            
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }
        
        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m, max_iter_elements = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm, max_iter_elements = self.vae.encode(feats_rst, lengths)
                    
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set

    def t2m_eval(self, batch):
        dev = batch["motion"].device
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        z_remo = None
        z_remo_text = None
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            text = batch["text"]
            if self.remo:
                # RETRIEVES THE MOTION
                B = len(text)
                all_indexes = []
                for b_ix in range(B): # 3.1sec per bs=32
                    length = int(lengths[b_ix])
                    batch_indexes = self.retrieve(text[b_ix], length, clip_model=None, device = dev)
                    all_indexes.extend(batch_indexes)
                all_indexes = np.array(all_indexes)
                all_motions = torch.Tensor(self.motions[all_indexes]).to(dev) # [batch, n_frames, 263]
                all_m_lengths = torch.Tensor(self.m_lengths[all_indexes]).long() # [batch]
                #all_captions = self.captions[all_indexes].tolist() # [batch, 5]
                
                # Encode retrieved motion with VAE
                z_remo, _, _ = self.vae.encode(all_motions[:,:torch.max(all_m_lengths)], all_m_lengths) # [5, B*num_retrieval, 256]
                #if z_remo.shape[1] != B:
                #    z_remo = z_remo.reshape(-1, B, z_remo.shape[2])

                # Encode retrieved captions with text_encoder
                #z_remo_text = self.text_encoder(all_captions) # [B*num_retrieval, 1, 768] #! REMOVE FO REBUTTAL

            

            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens

            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths, remo_cond=z_remo if z_remo!=None else None,)
                                            #remo_text_cond=z_remo_text if z_remo_text!=None else None) #! REMOVE FO REBUTTAL
            



        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:

                if self.subphase!='stage2' or self.subphase!='stage1':
                    z, dist_m, max_iter_elements = self.vae.encode(motions, lengths)
                else:
                    z_final = torch.tensor([], device=motions.device)
                    # Pass through the motion encoder self.nframes at time
                    for i in range(0, motions.shape[1], self.nframes):
                        if i + self.nframes > motions.shape[1]:
                            break
                        new_lenghts = torch.tensor([self.nframes] * motions.shape[0], device=motions.device)
                        z, dist_m, max_iter_elements = self.vae.encode(motions[:, i:i+self.nframes], new_lenghts)
                        z_final = torch.cat((z_final, z), dim=0) if z_final.shape[0] != 0 else z
                    
                    # Calcola resto
                    rest = motions.shape[1] % self.nframes
                    if rest != 0:
                        new_lenghts = torch.tensor([rest] * motions.shape[0], device=motions.device)
                        z, dist_m, max_iter_elements = self.vae.encode(motions[:, -rest:], new_lenghts)
                        z_final = torch.cat((z_final, z), dim=0) if z_final.shape[0] != 0 else z
                    
                    z = z_final
                    

                if self.save_latents:
                    new_batch = {}
                    new_batch['latent'] = z
                    new_batch['length'] = lengths
                    new_batch['text'] = texts
                    new_batch['motion'] = motions
                    new_batch['word_embs'] = word_embs
                    new_batch['pos_ohot'] = pos_ohot
                    new_batch['text_len'] = text_lengths

                    # See number of elements in folder
                    files = os.listdir('./datasets/latents')
                    latent_files = [f for f in files if 'latent' in f]
                    num_files = len(latent_files)

                    # Save latent with 6 digits
                    np.save(pjoin('./datasets/latents', f'latent_{num_files:06d}.npy'), z.cpu().numpy())

                
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                if self.subphase!='stage2' or self.subphase!='stage1':
                    feats_rst = self.vae.decode(z, lengths) # feats_rst is [bs, nframes, nfeats], with nframes = max(lengths)
                else:
                    feats_rst = torch.tensor([], device=z.device)
                    # Pass through the motion decoder
                    for i in range(0, z.shape[0]):
                        new_lenghts = torch.tensor([self.nframes] * z.shape[1], device=z.device)
                        feats = self.vae.decode(z[i].unsqueeze(0), new_lenghts) # feats is [batch_size, latent_dim[0], latent_dim[1]]
                        feats_rst = torch.cat((feats_rst, feats), dim=1) if feats_rst.shape[0] != 0 else feats
                  
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            
        # end time
        end = time.time()

        # ADDED BY ME
        # Remove extra frames and replace them with zeros in order to match with gt length
        max_len_batch = max(lengths)
        feats_rst_new = torch.zeros((feats_rst.shape[0], max_len_batch, feats_rst.shape[2]))
        for i in range(len(lengths)):
            if lengths[i] <= len(feats_rst[i]): # If the gt length of the motion is less than the max length of the batch
                feats_rst_new[i, :lengths[i]] = feats_rst[i][:lengths[i]]
            else:
                to_pad = lengths[i] - len(feats_rst[i])
                # Repeat the last frame to pad
                feats_rst_new[i, :lengths[i]] = torch.cat((feats_rst[i], feats_rst[i][-1].repeat(to_pad, 1)), dim=0)
        feats_rst = feats_rst_new.to(feats_rst.device)

        #if feats_rst.shape[1] > motions.shape[1]:
        #    feats_rst = feats_rst[:, :motions.shape[1], :]
        
 
        self.times.append(end - start)

        #print(feats_rst.shape)
        #print(motions.shape)
        # joints recover
        feats_rst = feats_rst
        motions = motions
        
        # HERE

        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)


        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]
        
        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set

    def a2m_eval(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            cond_emb = torch.cat((torch.zeros_like(actions), actions))

        if self.stage in ['diffusion', 'vae_diffusion']:
            z = self._diffusion_reverse(cond_emb, lengths)

        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert","actor"]:
                z, dist_m, max_iter_elements = self.vae.encode(motions, lengths)
            else:
                raise TypeError("vae_type must be mcross or actor")

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        mask = batch["mask"]
        joints_rst = self.feats2joints(feats_rst, mask)
        joints_ref = self.feats2joints(motions, mask)
        joints_eval_rst = self.feats2joints_eval(feats_rst, mask)
        joints_eval_ref = self.feats2joints_eval(motions, mask)

        rs_set = {
            "m_action": actions,
            "m_ref": motions,
            "m_rst": feats_rst,
            "m_lens": lengths,
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "joints_eval_rst": joints_eval_rst,
            "joints_eval_ref": joints_eval_ref,
        }
        return rs_set

    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")
            

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")
            
        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            if self.condition in ['text', 'text_uncond']:
                # use t2m evaluators
                rs_set = self.t2m_eval(batch)
            elif self.condition == 'action':
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit",
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(rs_set["m_action"],
                                                 rs_set["joints_eval_rst"],
                                                 rs_set["joints_eval_ref"],
                                                 rs_set["m_lens"])
                elif metric == "UESTCMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_lens"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        return loss
