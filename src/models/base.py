import torch
from torch import nn

def create_gridworld_convs(n_channels=1):
  return nn.Sequential(
        nn.Conv2d(n_channels, 8, 3, 1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1),
        nn.ReLU(),
        nn.Flatten()
    )

def create_atari_convs(n_channels=4):
  return nn.Sequential(
        nn.Conv2d(n_channels, 32, 5, 5, 0),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, 5, 0),
        nn.ReLU(),
        nn.Flatten()
    )