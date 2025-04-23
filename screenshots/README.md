Technical insights progression:
Experiment 1: use somewhat high LR (0.01)
- LR is low enough that training loss doesn't wiggle but high enough for validation loss to wiggle.
- Without using patchify, linear projection in TokenMixing = HxW-DWConv-strideHxW (or 1x1-Pointwise Conv along token dimension) => Massive shift toward global receptive field (long-range dependency), which means flatter loss landscape but has shit tons of saddle points according to [How Do Vision Transformers Work](https://openreview.net/forum?id=D78Go4hVcxO) spotlight paper.
![image](https://github.com/Skimmable-Code-pls/MLPMixer_numpy/blob/main/screenshots/MLPMixer48_patch1_depth10_epoch34.png)


Experiment 2: Use patch=14, Cosine scheduler with somewhat high LR (0.01) and linear warmup for 10 epochs
- LR is low enough that training loss doesn't wiggle but very high enough for validation loss to wiggle a lot.
- Warmup helps gradient descent plummet more during initial 2-3 epochs then it loses its usefulness.


Experiment 3: Use patch=3, Cosine scheduler with very high LR (0.1) and high warm restart to escape saddle point
- LR is low enough that training loss doesn't wiggle but very high enough for validation loss to wiggle a lot.
- Warm restart doesn't help escape saddle point but undo the gradient descent as seen in training loss.
