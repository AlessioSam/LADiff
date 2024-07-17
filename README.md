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


<h3 align="center"> 
<b>Download the checkpoints <a href="https://drive.google.com/drive/folders/1BFSzG3MdabhTydd27HvLleNh1n-fmsvH?usp=sharing">here</a>!</b>
</h3>

<h2 align="center">:bricks: REPO UNDER CONSTRUCTION</h2> 


Our code is heavily adapted from [MLD](https://github.com/ChenFengYe/motion-latent-diffusion). 
