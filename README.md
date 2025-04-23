TokenMixing = x2 linear projection of x^t with GELU and DropOut(optional) between 2 linear layers; ChannelMixing: x2 linear projection of x with GELU and DropOut(optional) between 2 linear layers. As linear projection of x^2 = PxP-DWConv-strideP, linear projection of x = 1x1-Pointwise Conv-stride1 [so MLPMixer is just convolution as Yann Lecunn puts it](https://x.com/ylecun/status/1390543133474234368?lang=en)? Not quite, 3x3-Conv-stride1 = 3x3-DWConv-stride1 -> 1x1-Pointwise Conv-stride1 but you clearly see there's a PxP-DWConv-strideP component so they can't be the same. Likewise, despite almost similar appearance between ConvNeXt block and MLPMixer as shown below, each of them will have a different profile on Hessian Eigenvalues-plot because you can't equate 3x3-DWConv-stride1 into a linear projection. So while **ResNet has negligible < 0 Hessian Eigenvalues and high magnitude Hessian Eigenvalues, ViT has marginal < 0 Hessian Eigenvalue and less-high magnitude Hessian Eigenvalues thanks to its Data Specificity** (see [How Vision Transformers work page 6](https://openreview.net/forum?id=D78Go4hVcxO)); **it's likely that ConvNeXt will sit inbetween ResNet and ViT but leaning toward ResNet and MLP-Mixer will sit outside of ViT as MLP-Mixer doesn't have Data Specificity like ViT**. ***TLDR: MLP-Mixer is a bad architecture***
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer_is_badConvNeXt.png)
 
Long epoch training MLPMixer on UNSW-NB15 dataset (100 epochs). Pretrained weight: mlpmixer198_patch14_embeddim198_depth10.npy
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer198_patch14_depth10_epoch100.png) <br>

Read technical key insights that I crafted from experiments in [screenshots/README.md](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/README.md). Here, I'll list only actionable ideas below to build from scratch using only numpy, scipy, math. Note that if * means I have successfully built it but it had adverse effects from experimenting or applying the principles from [How Do Vision Transformers Work?](https://openreview.net/forum?id=D78Go4hVcxO) here. If these ideas works out, I will try on tensorflow Vision Transformer. Finally, read [How Do Vision Transformers Work?](https://openreview.net/forum?id=D78Go4hVcxO) to understand how I came up with these actionable ideas <br>
- [ ] Add more Conv block at beginning following Alter-ResNet design in [How Do Vision Transformers Work?](https://openreview.net/forum?id=D78Go4hVcxO): Let's start ConvNeXt block first since it's easy to tweak MLP-Mixer block to get ConvNeXt; then ResNeXt or ResNet blocks as long as either of them have DW-Separable conv design from Efficient.
- [ ] random_masking from MAE. Start by reverse-engineering [img2img MLP-Mixer](https://github.com/MLI-lab/imaging_MLPs); then look at where masking module is placed [time-series MLP-Mixer x MAE]; then look at how MAE removes tokens with random_masking in encoder and [filling-in removed tokens at original positions with id_restores](https://github.com/facebookresearch/mae/blob/main/models_mae.py#L172-L196). We diverge from time-series MLP-Mixer x MAE because we don't intend to share MLP-Mixer block channelwise like they did.
- [ ] [Sparse SAM](https://github.com/jjsrf/SSAM-NEURIPS2024) as initial long-epoch experiments make me suspicious of non-convex loss landscape from looking at training loss around epoch 60 -> 100
- [ ] Add Gaussian noise to parameter update in optimiser.step(). Take inspiration from EDM2
- [ ] Inception Block and PatchEmbedding from [CT-img2img MLP-Mixer](https://arxiv.org/pdf/2402.17951). The Inception block they use is already intended for dimension reduction per [Inception block original paper](https://arxiv.org/pdf/1409.4842) and so as PatchEmbedding. If using 2 dimension reduction modules next to each other improved the performance, then how about 3 dimension reduction modules: Inception block -> PatchEmbedding -> random_masking.
- [x] Linear layer
- [x] Init linear layer's weight & bias w/ Kaiming uniform distribution
- [x] DropOut
- [x] LayerNorm
- [x] GeLU
- [ ] Tanh*. I won't post experiment because I couldn't be bother to train 100 epochs when initial 20 epochs is straight up bad compared to the worst configuration in here.
- [x] TokenMixer block*
- [x] ChannelMixer block
- [x] Adam optimiser w/ β1 = 0.9, β2 = 0.999, weight decay = 0, lr=0.001. Updating weight is the hardest bit to build and debug that I ended up accidentally find a way to consistently detect vanishing gradient within 3 epochs
- [x] Binary Cross Entropy with logits
- [x] Optimiser.zero_grad()
- [x] Warmup (epoch 0: lr=0 -> epoch 10: lr=0.001)
- [x] CosineAnnealingLR
- [x] CosineAnnealingLR with WarmRestarts*
- [x] StochasticDepth
- [ ] Set numpy seed to rule out lucky weight & bias init
- [ ] Gradient accumulation for more desired batch size = 4096 and reduce code overhead. For the moment, this is low priority because it doesn't make sense to accumulate for desired batch size = 4096 when the training dataset only has 20000 samples, and doing this requires me to rewrite def backward of every layer and activation function to incorporate gradient accumulation.
