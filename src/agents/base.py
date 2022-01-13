#import abc and abstractmethod
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from ..envs import TransitionData


class BaseAgent(ABC):
  @abstractmethod
  def sample_act(self, obs):
    pass

  def process_step_data(self, data: TransitionData):
    pass

  def end_step(self):
    pass

  def end_episode(self):
    pass

  def start_task(self, n_steps):
    pass

  def end_task(self):
    pass


class BaseRepresentationLearner(ABC):
  def __init__(self, model=None, batch_size=32, update_freq=32, log_freq=100):
    if model is None:
      self._init_model()
    else:
      self.model = model

    assert hasattr(self.model, 'encoder'), \
      'Model must have an encoder!'

    self.encoder = self.model.encoder
    self.batch_size = batch_size
    self.update_freq = update_freq
    self.log_freq = log_freq

  @abstractmethod
  def _init_model(self, *args, **kwargs):
    pass

  @abstractmethod
  def calculate_losses(self, batch_data):
    pass

  @abstractmethod
  def train(self, batch_data):
    pass


class ExperienceBufferMixin():
  def __init__(self, max_size=int(1e6)):
    self.exp_buffer = []
    self.max_size = max_size

  def _fix_size(self):
    self.exp_buffer = self.exp_buffer[-self.max_size:]

  def append_buffer(self, data):
    self.exp_buffer.append(data)
    self._fix_size()

  def extend_buffer(self, data):
    self.exp_buffer.extend(data)
    self._fix_size()

  def sample_buffer(self, n, replace=False):
    # Sample indices
    data_idxs = np.random.choice(range(len(self.exp_buffer)),
                                 size=n, replace=replace)
    batch_data = []
    for i in data_idxs:
      batch_data.append(self.exp_buffer[i])
    # Create separate np arrays for each element
    element_tensors = \
      [torch.stack([torch.tensor(se, dtype=torch.float32) for se in e], \
        dim=0) for e in zip(*batch_data)]

    return element_tensors

  def get_buffer_recent_data(self, n):
    batch_data = self.exp_buffer[-n:]
    # Create separate np arrays for each element
    element_tensors = \
      [torch.stack([torch.tensor(se, dtype=torch.float32) for se in e], \
        dim=0) for e in zip(*batch_data)]
    return element_tensors
  
  def buffer_size(self):
    return len(self.exp_buffer)


def create_basic_fe_model(layer_type='conv', input_dim=None):
  """
  Initializes a basic feature extractor
  
  Args:
    layer_type: 'conv' or 'linear'
    input_dim: (int) size of input (or number of channels)
    
  Returns:
    model: torch.nn.Module
  """
  if layer_type == 'linear':
    return nn.Sequential(
      nn.Linear(input_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64))
  elif layer_type == 'conv':
    return nn.Sequential(
          nn.Conv2d(input_dim, 8, 4, 2),
          nn.ReLU(),
          nn.Conv2d(8, 16, 3, 1),
          nn.ReLU(),
          nn.Flatten())
  raise Exception('Invalid layer type!')