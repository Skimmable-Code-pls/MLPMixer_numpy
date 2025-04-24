Technical insights progression: <br>
Initial impression: UNSW-NB15 is a tabular dataset where each feature column is in its own very, very distinct range: Some in [0; 127], others in [2.3+E09; 9.7+E09]. Even after rescale to [0; 1] and normalise, some column feature ends up being in [0; +-0.5] and other feature column have magnitude greater than 2. <br>

Experiment 1: use somewhat high LR (0.01), embed_dim=48, depth=10, patch=1
- LR is low enough that training loss doesn't wiggle but high enough for validation loss to wiggle.
- Without using patchify (patch=1), linear projection on x^T in TokenMixing = HxW-DWConv-strideHxW (or 1x1-Pointwise Conv along token dimension) => Massive shift toward global receptive field (long-range dependency), which means flatter loss landscape but has shit tons of saddle points according to [How Do Vision Transformers Work](https://openreview.net/forum?id=D78Go4hVcxO) spotlight paper.
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer48_patch1_depth10_epoch34.png)

Experiment 2: Use patch=14, Cosine scheduler with somewhat high LR (0.01) and linear warmup for 10 epochs
- LR is low enough that training loss doesn't wiggle but very high enough for validation loss to wiggle a lot.
- Warmup helps gradient descent plummet more during initial 2-3 epochs then it loses its usefulness.
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer198_patch14_depth10_epoch100.png)

Experiment 3: Use patch=3, Cosine-warm restart scheduler with very high LR (0.1) to escape saddle point
- LR is low enough that training loss doesn't wiggle but very high enough for validation loss to wiggle a lot.
- Warm restart doesn't help escape saddle point but undo the gradient descent as seen in training loss.
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer198_patch3_depth10_epoch100_schedulerWarmRestart_highLR.png)

Experiment 4: Cosine-warm restart scheduler with very low LR (0.0001) to escape saddle point but not too chaotic
- Warm restart is just shite that not even low LR can help it.
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer198_patch3_depth10_epoch100_schedulerWarmRestart_lowLR.png)


Experiment 5: Scrap warm restart, use normal Cosine schedule with very high LR for 2 epochs to boost gradient descent in initial 2 epochs but use sharp Cosine decay to prevent wiggle validation loss from epoch 3 onward. It works, look at the progression from epoch 3 onward, a lot smoother. However, the model doesn't learn much because my default eta_min for cosine scheduler turns out to be too small in this scenario, which is sad considering how flat MLPMixer's loss landscape is: So it seems like I should boost eta_min to 4e-5 and add Conv block.
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/mlpmixer198_patch3_embeddim198_depth10_sharpCosine.npy)
