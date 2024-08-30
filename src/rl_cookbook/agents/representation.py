import copy
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from .base import BaseRepresentationLearner
from ..envs import TransitionData


class NextStatePredictor(BaseRepresentationLearner):
  """Learns representations by predicting the next state given the current state and action."""

  def __init__(
      self,
      model: nn.Module,
      n_acts: int,
      batch_size: int = 256,
      update_freq: int = 128,
      lr: float = 1e-3):
    """
    Args:
      model: The neural network model for prediction.
      n_acts: Number of possible actions.
      lr: Learning rate for the optimizer.
    """
    super().__init__(model, batch_size, update_freq)
    self.encoder = self.model.encoder
    self.n_acts = n_acts
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
  def _init_model(self):
    raise Exception('Next state prediction requires a model to be specified!')

  def calculate_losses(self, batch_data):
    """
    Calculate losses for a batch of data.

    Args:
      batch_data: A batch of transition data.

    Returns:
      torch.Tensor: The calculated losses.
    """
    device = next(self.model.parameters()).device
    
    obs, acts, _, next_obs, _ = \
      [torch.tensor(e, dtype=torch.float32).to(device) for e in batch_data]
    oh_acts = F.one_hot(acts.long(), self.n_acts).float()
    next_obs_pred = self.model(obs, oh_acts)
    losses = (next_obs - next_obs_pred) ** 2
    while len(losses.shape) > 2:
      losses = losses.mean(dim=-1)
    
    return losses

  def train(self, batch_data):
    """
    Train the model on a batch of data.

    Args:
      batch_data: A batch of transition data.

    Returns:
      float: The mean loss for this batch.
    """
    losses = self.calculate_losses(batch_data)
    loss = losses.mean()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

    
class SFPredictor(BaseRepresentationLearner):
  """Learns representations using Successor Feature prediction."""

  def __init__(
      self,
      model: nn.Module,
      batch_size: int = 256,
      update_freq: int = 128,
      log_freq: int = 100,
      target_net_update_freq: int = 64,
      discount_factor: float = 0.99,
      lr: float = 1e-3):
    """
    Args:
      model: The neural network model for prediction.
      target_net_update_freq: Frequency of target network updates.
      discount_factor: Discount factor for future rewards.
      lr: Learning rate for the optimizer.
    """
    super().__init__(model, batch_size, update_freq, log_freq)

    self.discount_factor = discount_factor
    self.target_net_update_freq = target_net_update_freq
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self._update_target_model()
    self.train_step_idx = 0

  def _init_model(self):
    raise Exception('Next state prediction requires a model to be specified!')

  def _update_target_model(self):
    """Update the target model with the current model's parameters."""
    self.target_model = copy.deepcopy(self.model)
    for param in self.target_model.parameters():
      param.requires_grad = False

  def calculate_losses(self, batch_data):
    """
    Calculate losses for a batch of data using Successor Feature prediction.

    This method computes the loss between predicted successor features and target successor features.
    The target successor features are calculated using the target network and the discount factor.

    Args:
      batch_data: A batch of transition data containing observations, actions, rewards, next observations, and done flags.

    Returns:
      torch.Tensor: The calculated losses for each sample in the batch.
    """
    device = next(self.model.parameters()).device

    obs, _, _, next_obs, _ = \
      [torch.tensor(e, dtype=torch.float32).to(device) for e in batch_data]
    
    self.model.train()
    self.target_model.train()

    with torch.no_grad():
      belief_state = self.target_model.encoder(obs)
      _, next_sfs = self.target_model(next_obs)
      # TODO: Add in dones
      target_sfs = belief_state + self.discount_factor * next_sfs

    # TODO: Add policy as input to this model
    _, sf_preds = self.model(obs)

    losses = torch.sum((target_sfs - sf_preds) ** 2, dim=-1)

    return losses

  def train(self, batch_data):
    """
    Train the Successor Feature predictor model on a batch of data.

    This method performs one step of training on the model using the provided batch of transition data.
    It calculates the losses, performs backpropagation, updates the model parameters, and periodically
    updates the target network.

    Args:
      batch_data: A batch of transition data containing observations, actions, rewards, next observations, and done flags.

    Returns:
      float: The mean loss for this batch, which can be used for monitoring training progress.
    """
    losses = self.calculate_losses(batch_data)
    loss = losses.mean()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    if (self.train_step_idx + 1) % self.target_net_update_freq == 0:
      self._update_target_model()

    self.train_step_idx += 1

    return loss.item()