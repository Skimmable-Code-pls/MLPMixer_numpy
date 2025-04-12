Report after training MLP-Mixer on UNSW-NB15 dataset. Pretrained weight is included here under .npy file. You should use the model and optimiser state loading function that I wrote in the MLP-Mixer_for_1dim-input.ipynb ![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/MLP-Mixer_depth10_48_embed_dim.png)

Some key insights I discover in making MLP-Mixer work with UNSW-NB15, a non-image specific dataset where every pixel is in extreme range from 0->9.57+E09
- Avoid using any data augmentation. No point of augmenting dataset if the augmented version doesn't bear any coherence.
- Like figure 4 in [this paper](https://arxiv.org/pdf/2106.10270), the best model size without tends to be small-but-not-too-small and mid-size. The trick is to find what is too-small model and too-big model for any given dataset. I simply sweep embedding dimension for the same depth until I find one with ***promising training loss and validation loss drop in the initial first 150 iterations***. Once I find one, I increase the depth.
- To detect vanishing gradient is to monitor max absolute difference of a linear layer weight before and after optimiser step throughout training. If it is approach 0 quickly yet loss stays the same, then it is vanishing gradient.
- To augment unknown dataset, 2 good ideas I would like to try.
   1. Train VAE to map NIDS dataset into latent space like how VAE is used prior to training Stable Diffusion, with dataset mapped to latent space I feel safe to add random noise to augment the dataset.
   2.  Inspired from MAE, mask random 'patch' for each training batch to a more reduced size along sequence dimension.
- The hardest bits to implement in numpy are the most mundane bits in Pytorch: optimizer.step() and optimizer.zero_grad(). Fortunately, as I have implemented backward pass for each layer manually, I can pin point exactly which module attribute to empty dict or list to prevent memory leak.
- Permutation invariance isn't as big of a deal as I have originally thought given this project and [someone's else project on MLP-Mixer](https://github.com/sijan67/Exploring-the-MLP-Mixer-Architecture/tree/main). However, this maybe just means that when the resolution is very small and the amount of redundant pixel is very very small, the adverse effect of permutation invariance isn't amplified as it would be have if the resolution was 224x224 for example.

To-Do
- [ ] Warmup epoch entirely without using torch
- [ ] Lr scheduler without using torch
- [ ] Build MAE from scratch without using torch and add-on MLP-Mixer backbone to counter overfitting problem 
