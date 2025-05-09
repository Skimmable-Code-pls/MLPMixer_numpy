# Overview
This repo is to achieve the following:
- [x] Build ML model from scratch 99.9% in numpy
- [x] Preprocessing tabular dataset where every spatial feature has completely different "pixel magnitude" e.g. in UNSW-NB15 dataset, some column has pixen magnitude in range of 0->150 while other in range of 2.3+E9->9+E9. Currently I'm forced to use only rescaling and normalisation, but converting pixel space into latent space and use it alongside masked autoencoder is [an interesting prospect](https://arxiv.org/pdf/2502.03444) for future work
- [ ] Fix architectural problem of a given neural network to improve model accuracy

# Background
TokenMixing = double linear projection of x^t with GELU and DropOut(optional) between 2 linear layers <br>
ChannelMixing = double linear projection of x with GELU and DropOut(optional) between 2 linear layers <br>
As linear projection of x^2 = PxP-DWConv-strideP, linear projection of x = 1x1-Pointwise Conv-stride1 [so MLPMixer is just convolution as Yann Lecunn puts it](https://x.com/ylecun/status/1390543133474234368?lang=en) and moreover looks like a variant of ConvNeXt as shown below? <br>
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer_is_badConvNeXt.png)
Not quite: 3x3-Conv-stride1 = 3x3-DWConv-stride1 -> 1x1-Pointwise Conv-stride1. However, PxP-DWConv-strideP isn't 3x3-DWConv-stride1 with kernel size P, the key factor is having kernel size < stride so PxP-DWConv-strideP is more akin to Pointwise Conv and linear projection than the standard Conv nor DWConv: Meaning PxP-DWConv-strideP will share the same long-range dependency issue like MSAttns but non of the Data Specificity like MSAttns as shown in [How Vision Transformers work page 6](https://openreview.net/forum?id=D78Go4hVcxO). If I am to put MLPMixer on the loss landscape's spectrum using the insights learnt from then it will look as below. **TLDR: MLP-Mixer is a shite architecture so any idea that improve it on small tabular dataset will also be an improvement on current VLM and LLM.** <br>
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/loss_landscape_spectrum.png)


# Experiment
Read experiment nodes in [screenshots/README.md](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/README.md). Here, I'll list only actionable ideas below to build from scratch using only numpy, scipy, math. Note that * means the idea worsen the model's performance according to my experiments or [How Do Vision Transformers Work?](https://openreview.net/forum?id=D78Go4hVcxO). All my ideas are based on the loss landscape spectrum that I've created above, basically I want to bring it closer to the middle where Swin is at, since there's only so much I can do to modify the architecture, I have to find other ideas to augment dataset, change optimiser and use a Diffusion's noise scheduler in optimiser.step() to deal with MLPMixer's loss landscape.
- [ ] Add more Conv block at beginning of every depth and more blocks at early depth following Alter-ResNet design in [How Do Vision Transformers Work?](https://openreview.net/forum?id=D78Go4hVcxO): Let's start ConvNeXt block first since it's easy to tweak MLP-Mixer block to get ConvNeXt; then ResNeXt or ResNet blocks as long as either of them have DW-Separable conv design from Efficient. Now, because looping convolution will get slow very quickly as depth scales up from 1 -> 2, I need to convolution into dotproduct by reverse-engineer [img2col and col2img](https://github.com/lujiazho/MachineLearningPlayground/blob/main/Tutorials/CNN/CNN_img2col_numba.ipynb). When Conv is implemented, don't use Patchify.
- [ ] random_masking from MAE. Start by reverse-engineering [img2img MLP-Mixer](https://github.com/MLI-lab/imaging_MLPs); then look at where masking module is placed [time-series MLP-Mixer x MAE]; then look at how MAE removes tokens with random_masking in encoder and [filling-in removed tokens at original positions with id_restores](https://github.com/facebookresearch/mae/blob/main/models_mae.py#L172-L196). We diverge from time-series MLP-Mixer x MAE because we don't intend to share MLP-Mixer block channelwise like they did.
- [ ] [Sparse SAM optimiser](https://github.com/jjsrf/SSAM-NEURIPS2024) to flatten loss landscape, which I ***may*** need after I add more Conv layers to this architecture.
- [ ] Add Gaussian noise to parameter update in optimiser.step() per advised by [this recent paper about escaping saddle points](https://arxiv.org/pdf/2410.02017). Take inspiration from EDM2 to figure how to design a proper noise scheduler w.r.t training iterations
- [ ] [Lambda layer](https://arxiv.org/pdf/2102.08602). Hardest actionable idea but also maybe the most worth it because it basically has the Data Specificity benefit of MSAttns but none of its' issues that stemmed from long-range dependency.
- [ ] Inception Block and PatchEmbedding from [CT-img2img MLP-Mixer](https://arxiv.org/pdf/2402.17951). The Inception block can do some spatial smoothing, which is good according to [How Do Vision Transformers Work?](https://openreview.net/forum?id=D78Go4hVcxO).
- [x] Linear layer*
- [x] Init linear layer's weight & bias w/ Kaiming uniform distribution
- [x] DropOut
- [x] LayerNorm
- [x] GeLU
- [x] Tanh*. I won't post experiment because I couldn't be bother to train 100 epochs when initial 20 epochs is straight up bad compared to the worst configuration in here.
- [x] TokenMixer block*
- [x] ChannelMixer block
- [x] Adam optimiser w/ β1 = 0.9, β2 = 0.999, weight decay = 0, lr=0.001. Updating weight is the hardest bit to build and debug that I ended up accidentally find a way to consistently detect vanishing gradient within 3 epochs
- [x] Binary Cross Entropy with logits
- [x] Optimiser.zero_grad()
- [x] Model.zero_grad(). Because I'm doing backprop manually without the assistance of torch's Autograd, so I also have to delete each layer's gradient manually after backprop. 
- [x] Warmup (epoch 0: lr=0 -> epoch 10: lr=0.001)
- [x] "Sharpening" Cosine scheduler. It's improving the model's performance in the ongoing experiment so far
- [x] CosineScheduler with warm restart*
- [x] StochasticDepth
- [ ] Set numpy seed to rule out lucky weight & bias init
- [ ] Gradient accumulation for more desired batch size = 4096 and reduce code overhead. For the moment, this is low priority because it doesn't make sense to accumulate for desired batch size = 4096 when the training dataset only has 20000 samples, and doing this requires me to rewrite def backward of every layer and activation function to incorporate gradient accumulation.
