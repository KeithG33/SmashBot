# SmashBot

A Super Smash Bros. Melee AI, written in PyTorch. It learns to play by watching
hundreds of thousands of human games, then improves by playing against itself.

This is a PyTorch reimplementation of the approach pioneered by
[slippi-ai](https://github.com/vladfi1/slippi-ai) (Phillip II), with many changes coming from my own fun and experimentation. 


## Stage 1: Imitation learning

Behaviour cloning on a very large pile of anonymized ranked replays (thank you everyone), covering
twelve characters. The result is a fairly capable and recognisably human model, albeit with some silly blind spots and funny mistakes.

The model architecture was chosen through experiments on the imitation learning dataset, currently landing on an a-MLP architecture (6L / 576W) relying on Spatial Gating Blocks from [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050).

## Stage 2: Reinforcement learning

The imitation learning model is frozen as the teacher, and a copy of it becomes the student that trains with self-play and an opponent pool composed of past snapshots, previous best models, and different strengths of Phillip. The opponents are sampled using Prioritized Fictitious Self-Play (PFSP) from AlphaStar. A KL penalty against the teacher keeps it playing human Melee, and the league of opponents keeps the model from overfitting or cycling between strategies via self-play.

## Status

A research project, built for fun and for learning. Not affiliated with any of
the projects below.

## Acknowledgements

Mostly copying slippi-ai for this acknowledgement :D 

- Huge thanks to **vladfi1** for
  [slippi-ai](https://github.com/vladfi1/slippi-ai), which this project is built
  on in every sense: the two-stage recipe, the delay handling, the replay
  parsing, the Dolphin wrangling, much of the base code and functions used; if you are interested in Melee AI, go there first.
- Big thanks to **altf4** for [libmelee](https://github.com/altf4/libmelee), the
  interface to Slippi Dolphin, this is the backend API needed for training bots.
- HUGE thanks to **Fizzi** for Slippi, the community would be a wildly different place without it (and for the fast-forward Gecko code for training).
- Thanks to those who helped curate the replay dataset and those who donated replays.
- Thanks to KyleH for lightweight sim backend to speed up training and remove the reliance on the Dolphin emulator (and shout out to the decompilation project!!).
