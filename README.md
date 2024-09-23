<div align="center">
<h3>European Conference on Computer Vision 2024</h3>
<h1>Length-Aware Motion Synthesis via Latent Diffusion</h1>
<h3> <i>Alessio Sampieri*, Alessio Palma*, Indro Spinelli, and Fabio Galasso</i></h3>
 <h4> <i>Sapienza University of Rome, Italy</i></h4>
 
 [[Paper](https://arxiv.org/abs/2407.11532)]
 
<image src="https://github.com/AlessioSam/LADiff/blob/main/images/teaser-1.png" width="1000">
       
</div>



<h2 align="center">Abstract</h2> 
<div align="center"> 
<p>
The target duration of a synthesized human motion is a critical attribute that requires modeling control over the motion dynamics and style. Speeding up an action performance is not merely fast-forwarding it. However, state-of-the-art techniques for human behavior synthesis have limited control over the target sequence length.
  
We introduce the problem of generating length-aware 3D human motion sequences from textual descriptors, and we propose a novel model to synthesize motions of variable target lengths, which we dub ``Length-Aware Latent Diffusion'' (_LADiff_). _LADiff_ consists of two new modules: 1) a length-aware variational auto-encoder to learn motion representations with length-dependent latent codes; 2) a length-conforming latent diffusion model to generate motions with a richness of details that increases with the required target sequence length. _LADiff_ significantly improves over the state-of-the-art across most of the existing motion synthesis metrics on the two established benchmarks of HumanML3D and KIT-ML.
</p>
</div>

<hr/>

### Create the environment

```
conda create python=3.10 --name ladiff
conda activate ladiff
```

Install the packages in `requirements.txt` and install [PyTorch 1.12.1](https://pytorch.org/)

```
cd src
pip install -r requirements.txt
```

Run the scripts to download dependencies:

```
bash prepare/download_smpl_model.sh
bash prepare/prepare_clip.sh
bash prepare/download_t2m_evaluators.sh
```

Put datasets in the `datasets` folder, please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for setup.

We tested our code on Python 3.10.9 and PyTorch 1.12.1.

<hr/>

### Pretrained model

Download the checkpoints trained on HumanML3D from [the Google Drive](https://drive.google.com/drive/folders/1BFSzG3MdabhTydd27HvLleNh1n-fmsvH?usp=sharing), and place them in the `experiments/ladiff` folder. 

<hr/>

### Train your own model

For the stage 1 (LA-VAE) please first check the parameters in `configs/config_vae_humanml3d.yaml`, e.g. `NAME`,`DEBUG`.

Then, run the following command:

```
python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
```

For the stage 2 (LA-DDPM) please update the parameters in `configs/config_ladiff_humanml3d.yaml`, e.g. `NAME`,`DEBUG`,`PRETRAINED_VAE` (change to your latest ckpt model path in previous step)

Then, run the following command:

```
python -m train --cfg configs/config_ladiff_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 128  --nodebug
```

<hr/>

### Evaluate the model

Please first put the trained model checkpoint path to `TEST.CHECKPOINT` in `configs/config_ladiff_humanml3d.yaml`.

Then, run the following command:

```
python -m test --cfg configs/config_ladiff_humanml3d.yaml --cfg_assets configs/assets.yaml
```

<hr/>

### Citation

If you find our code or paper helpful, please consider citing us.

<hr/>

### Acknowledgements

Thanks to [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), our code is borrowing from them. Please visit their page for more instructions. 
