#import abc and abstractmethod
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from ..envs import TransitionData


class BaseAgent(ABC):
  """Abstract base class for reinforcement learning agents."""

  @abstractmethod
  def sample_act(self, obs):
    """Samples an action given an observation."""
    pass

  def process_step_data(self, data: TransitionData):
    """Processes data from a single environment step."""
    pass

  def end_step(self):
    """Called at the end of each environment step."""
    pass

  def end_episode(self):
    """Called at the end of each episode."""
    pass

  def start_task(self, n_steps):
    """Called at the start of a new task."""
    pass

  def end_task(self):
    """Called at the end of a task."""
    pass


class BaseRepresentationLearner(ABC):
  """Abstract base class for representation learning models."""

  def __init__(self, model=None, batch_size=32, update_freq=32, log_freq=100):
    """
    Args:
      model: The model to use for representation learning.
      batch_size: Size of batches for training.
      update_freq: Frequency of model updates.
      log_freq: Frequency of logging.
    """
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
    """Initializes the model."""
    pass

  @abstractmethod
  def calculate_losses(self, batch_data):
    """Calculates losses for a batch of data."""
    pass

  @abstractmethod
  def train(self, batch_data):
    """Trains the model on a batch of data."""
    pass


class ExperienceBufferMixin():
  """Mixin class for experience replay buffer functionality."""

  def __init__(self, max_size=int(1e6)):
    """
    Args:
      max_size: Maximum size of the buffer.
    """
    self.exp_buffer = []
    self.max_size = max_size

  def _fix_size(self):
    """Ensures the buffer does not exceed the maximum size."""
    self.exp_buffer = self.exp_buffer[-self.max_size:]

  def clear_buffer(self):
    """Clears the experience buffer."""
    self.exp_buffer = []

  def append_buffer(self, data):
    """Appends a single piece of data to the buffer."""
    self.exp_buffer.append(data)
    self._fix_size()

  def extend_buffer(self, data):
    """Extends the buffer with multiple pieces of data."""
    self.exp_buffer.extend(data)
    self._fix_size()

  def sample_buffer(self, n, replace=False):
    """
    Samples n experiences from the buffer.

    Args:
      n: Number of samples to draw.
      replace: Whether to sample with replacement.

    Returns:
      A list of tensors, each containing a specific element of the sampled experiences.
    """
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
    """
    Retrieves the n most recent experiences from the buffer.

    Args:
      n: Number of recent experiences to retrieve.

    Returns:
      A list of tensors, each containing a specific element of the recent experiences.
    """
    batch_data = self.exp_buffer[-n:]
    # Create separate np arrays for each element
    element_tensors = \
      [torch.stack([torch.tensor(se, dtype=torch.float32) for se in e], \
        dim=0) for e in zip(*batch_data)]
    return element_tensors
  
  def buffer_size(self):
    """Returns the current size of the buffer."""
    return len(self.exp_buffer)


def create_basic_fe_model(layer_type='conv', input_dim=None):
  """
  Initializes a basic feature extractor model.
  
  Args:
    layer_type: Type of layers to use ('conv' or 'linear').
    input_dim: Size of input (or number of channels for conv layers).
    
  Returns:
    A PyTorch Sequential model.

  Raises:
    Exception: If an invalid layer type is specified.
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