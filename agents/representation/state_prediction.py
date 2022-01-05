import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from ..base import BaseAgent, BaseRepresentationLearner
from ...envs import TransitionData


class NextStatePredReprLearner(BaseRepresentationLearner):
  def __init__(
      self,
      model: nn.Module,
      lr: float = 1e-3):
    super().__init__(model)
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
  def _init_model(self):
    raise Exception('Next state prediction requires a model to be specified!')

  def train(self, batch_data: list[TransitionData]):
    device = next(self.model.parameters()).device
    
    obs, acts, _, next_obs, _ = \
      [torch.stack([torch.tensor(se, dtype=torch.float32) for se in e], \
        dim=0).to(device) for e in zip(*batch_data)]

    next_obs_pred = self.model(obs, acts)
    loss = F.mse_loss(next_obs_pred, next_obs)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    print(loss.item())