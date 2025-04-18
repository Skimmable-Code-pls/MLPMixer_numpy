Report after training MLP-Mixer on UNSW-NB15 dataset. Pretrained weight is included here under .npy file. You should use the model and optimiser state loading function that I wrote in the MLP-Mixer_for_1dim-input.ipynb <br> ![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/MLP-Mixer_depth10_48_embed_dim.png)

Some key insights I discover in making MLP-Mixer work with UNSW-NB15, a non-image specific dataset where every pixel is in extreme range from 0->9.57+E09
- Avoid using any data augmentation. No point of augmenting dataset if the augmented version doesn't bear any coherence. Which brings me 2 ideas to improve Network Intrusion Detection model, either add-on random_masking to resize size along sequence dimension inspired from MAE; or build VAE from scratch and train it or use my pretrained MAE in this project to map normalised dataset into latent like how VAE is used as prior to training Stable Diffusion then add random noise to latent.
- Like figure 4 in [this paper](https://arxiv.org/pdf/2106.10270), the best model size without tends to be small-but-not-too-small and mid-size. The trick is to find what is too-small model and too-big model for any given dataset. I simply sweep embedding dimension for the same depth until I find one with ***promising training loss and validation loss drop in the initial first 150 iterations***. Once I find one, I increase the depth.
- To detect vanishing gradient is to monitor max absolute difference of a linear layer weight before and after optimiser step throughout training. If it is approach 0 quickly yet loss stays the same, then it is vanishing gradient.
-   Contrary to what I originally worried about GELU activation function having an impact in creating vanishing gradient due to >90% 'pixels' of normalised UNSW-NB15 dataset within the range [-2.5; 0], it turnt out that replacing GELU with Tanh just made things worse, which it shouldn't have been had my worry was correct. So if it's not activation function, then tweaking weight decay and learning rate are so far the only way to deal with vanishing gradients.
- Permutation invariance isn't as big of a deal as I have originally thought given this project and [someone's else project on MLP-Mixer](https://github.com/sijan67/Exploring-the-MLP-Mixer-Architecture/tree/main). However, this maybe just means that when the resolution is very small and the amount of redundant pixel is very very small, the adverse effect of permutation invariance isn't amplified as it would be have if the resolution was 224x224 for example.

To Build From Scratch without using torch
- [ ] Set a simple condition to activate weight decay with following heuristics: (epoch 3: weight decay=0.3 -> epoch 5: weight decay=0.1 -> epoch 7: weight_decay=0.03 -> epoch 9 onward: weight_decay=0.001)
- [ ] Warmup (epoch 0: lr=0.001 -> epoch 4: lr=0.01)
- [ ] CosineAnnealingWarmRestarts LR scheduler (T_0 = 4, eta_min=0.00001)
- [ ] Stochastic Depth(0.1)
- [ ] Gradient accumulation for more desired batch size = 4096 and reduce code overhead
- [ ] Build MAE from scratch then add-on MLP-Mixer backbone to counter overfitting problem. To adapt random_masking module, create a dropout mask with uniform distribution then elementwise-multiply with projected input; take caution in adapting id_restores. To adapt decoder, start by reverse-engineering [this code](https://github.com/facebookresearch/mae/blob/main/models_mae.py#L172-L196)
- [ ] Can use pretrained MAE or build and train VAE from scratch to create latent, then add random noise to that latent

Here are the stuffs I have built from scratch without using torch
- [x] Linear layer w/ manual backprop and init weight & bias w/ Kaiming uniform distribution
- [x] DropOut=0.1 w/ manual backprop
- [x] LayerNorm w/ manual backprop
- [x] GeLU or Tanh w/ manual backprop
- [x] TokenMixer block
- [x] ChannelMixer block
- [x] Adam optimiser w/ β1 = 0.9, β2 = 0.999, weight decay = 0.03, lr=0.003. Updating weight is the hardest bit to build and debug that I ended up accidentally find a way to consistently detect vanishing gradient within 3 epochs.
- [x] Binary Cross Entropy with logits
- [x] Optimiser.zero_grad()
