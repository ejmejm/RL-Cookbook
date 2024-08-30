import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Source: https://github.com/hwang-ua/fta_pytorch_implementation/blob/main/core/lta.py
class FTA(nn.Module):
  """Fixed Tile Activation (FTA) module.

  This module implements a fixed tile coding activation function.

  Args:
    input_dim: Dimension of the input.
    tiles: Number of tiles (default: 20).
    bound_low: Lower bound for tiling (default: -2).
    bound_high: Upper bound for tiling (default: 2).
    eta: Smoothing parameter (default: 0.2).
  """

  def __init__(self, input_dim, tiles=20, bound_low=-2, bound_high=2, eta=0.2):
    super(FTA, self).__init__()
    # 1 tiling, binning
    self.n_tilings = 1
    self.n_tiles = tiles
    self.bound_low, self.bound_high = bound_low, bound_high
    self.delta = (self.bound_high - self.bound_low) / self.n_tiles
    c_mat = torch.as_tensor(np.array([self.delta * i for i in range(self.n_tiles)]) + self.bound_low, dtype=torch.float32)
    self.register_buffer('c_mat', c_mat)
    self.eta = eta
    self.d = input_dim

  def forward(self, reps):
    """Forward pass of the FTA module.

    Args:
      reps: Input tensor.

    Returns:
      Tensor: Activated output.
    """
    temp = reps
    temp = temp.reshape([-1, self.d, 1])
    onehots = 1.0 - self.i_plus_eta(self.sum_relu(self.c_mat, temp))
    out = torch.reshape(torch.reshape(onehots, [-1]), [-1, int(self.d * self.n_tiles * self.n_tilings)])
    return out

  def sum_relu(self, c, x):
    """Computes the sum of ReLU activations."""
    out = F.relu(c - x) + F.relu(x - self.delta - c)
    return out

  def i_plus_eta(self, x):
    """Applies smoothing to the input."""
    if self.eta == 0:
      return torch.sign(x)
    out = (x <= self.eta).type(torch.float32) * x + (x > self.eta).type(torch.float32)
    return out