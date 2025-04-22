Short epoch training MLPMixer on UNSW-NB15 dataset (34 epochs). Pretrained weight for this report is model_optim_state.npy under pretrained_weights folder. Use model and optimiser state loading function that I wrote in the MLP-Mixer_for_1dim-input.ipynb <br> ![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLP-Mixer_depth10_48_embed_dim.png)

Some key insights I discover in making MLP-Mixer work with UNSW-NB15, a non-image specific dataset where every pixel is in extreme range from 0->9.57+E09
- UNSWNB15 is a non-image specific dataset the 'normalised pixel' in each column is a complete different range that can differ from [-2.5, 0] to [2.3+E9, 9.57+E9] so avoid MixUp and CutMix.
- To detect vanishing gradient is to monitor max absolute difference of a linear layer weight before and after optimiser step throughout training. If it is approach 0 quickly yet loss stays the same, then it is vanishing gradient.
- Contrary to what I originally worried about GELU activation function having an impact in creating vanishing gradient due to >90% 'pixels' of normalised UNSW-NB15 dataset within the range [-2.5; 0], it turnt out that replacing GELU with Tanh just made things worse, which it shouldn't have been had my worry was correct. So if it's not activation function, then tweaking weight decay and learning rate are so far the only way to deal with vanishing gradients.
- Permutation invariance isn't as big of a deal as I have originally thought given this project and [someone's else project on MLP-Mixer](https://github.com/sijan67/Exploring-the-MLP-Mixer-Architecture/tree/main). However, this maybe just means that when the resolution is very small and the amount of redundant pixel is very very small, the adverse effect of permutation invariance isn't amplified as it would be have if the resolution was 224x224 for example. Indeed, in more spotlight paper like [How do Vision Transformers work](https://openreview.net/forum?id=D78Go4hVcxO) 'permutation invariance' doesn't matter at all.

Long epoch training MLPMixer on UNSW-NB15 dataset (100 epochs)
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer198_patch14_depth10_epoch100.png) <br>

Here are the stuffs I have built from scratch without using torch
- [x] Linear layer w/ manual backprop and init weight & bias w/ Kaiming uniform distribution
- [x] DropOut=0.1 w/ manual backprop
- [x] LayerNorm w/ manual backprop
- [x] GeLU or Tanh w/ manual backprop
- [x] TokenMixer block
- [x] ChannelMixer block
- [x] Adam optimiser w/ β1 = 0.9, β2 = 0.999, weight decay = 0, lr=0.001. Updating weight is the hardest bit to build and debug that I ended up accidentally find a way to consistently detect vanishing gradient within 3 epochs
- [x] Binary Cross Entropy with logits
- [x] Optimiser.zero_grad()
- [x] Warmup (epoch 0: lr=0 -> epoch 10: lr=0.001)
- [x] CosineAnnealingLR scheduler (T_max = num_epochs, eta_min=0.00001)
- [x] Stochastic Depth(0.1)

To-do in numpy:. If this works out, I will try on tensorflow Vision Transformer.
- [ ] Adapt random_masking from MAE. Start by reverse-engineering [img2img MLP-Mixer](https://github.com/MLI-lab/imaging_MLPs); then look at where masking module is placed [time-series MLP-Mixer x MAE]; then look at how MAE removes tokens with random_masking in encoder and [filling-in removed tokens at original positions with id_restores](https://github.com/facebookresearch/mae/blob/main/models_mae.py#L172-L196). We diverge from time-series MLP-Mixer x MAE because we don't intend to share MLP-Mixer block channelwise like they did.
- [ ] [Sparse SAM](https://github.com/jjsrf/SSAM-NEURIPS2024) as initial long-epoch experiments make me suspicious of non-convex loss landscape from looking at training loss around epoch 60 -> 100
- [ ] Inception Block and PatchEmbedding from [CT-img2img MLP-Mixer](https://arxiv.org/pdf/2402.17951). The Inception block they use is already intended for dimension reduction per [Inception block original paper](https://arxiv.org/pdf/1409.4842) and so as PatchEmbedding. If using 2 dimension reduction modules next to each other improved the performance, then how about 3 dimension reduction modules: Inception block -> PatchEmbedding -> random_masking.
- [ ] Add Gaussian noise to parameter update


To-do misc:
- [ ] Set numpy seed to rule out lucky weight & bias init
- [ ] Gradient accumulation for more desired batch size = 4096 and reduce code overhead. For the moment, this is low priority because it doesn't make sense to accumulate for desired batch size = 4096 when the training dataset only has 20000 samples, and doing this requires me to rewrite def backward of every layer and activation function to incorporate gradient accumulation.
