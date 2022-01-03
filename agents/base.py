#import abc and abstractmethod
from abc import ABC, abstractmethod

from gym import spaces
import numpy as np
import torch

class BaseAgent(ABC):
  @abstractmethod
  def sample_act(self, obs):
    pass

  def process_step_data(self, data):
    pass

  def end_step(self):
    pass

  def end_episode(self):
    pass

  def start_task(self, n_steps):
    pass

  def end_task(self):
    pass


class ExperienceBufferMixin():
  def __init__(self, max_size=int(1e6)):
    self.buffer = []
    self.max_size = max_size

  def _fix_size(self):
    self.buffer = self.buffer[-self.max_size:]

  def append_data(self, data):
    self.buffer.append(data)
    self._fix_size()

  def extend_data(self, data):
    self.buffer.extend(data)
    self._fix_size()

  def sample(self, n, replace=False):
    # Sample indices
    data_idxs = np.random.choice(range(len(self.buffer)),
                                 size=n, replace=replace)
    batch_data = []
    for i in data_idxs:
      batch_data.append(self.buffer[i])

    # Create separate np arrays for each element
    batch_data = np.array(batch_data)
    element_tensors = \
      [torch.from_numpy(np.stack(batch_data[:, i])) \
      for i in range(batch_data.shape[1])]
    
    return element_tensors


class EzExplorerAgent(BaseAgent):
  def __init__(self, env, min_repeat=1, max_repeat=6):
    if type(env.action_space) != spaces.Discrete:
      raise Exception('EzExplorerAgent only supports discrete action spaces!')

    self.n_acts = env.action_space.n
    self.min_repeat = min_repeat
    self.max_repeat = max_repeat

    self.curr_act = None
    self.repeats_left = 0

  def sample_act(self):
    if self.repeats_left > 0:
      self.repeats_left -= 1
      return self.curr_act
    
    self.curr_act = np.random.randint(0, self.n_acts + 1)
    self.repeats_left = np.random.randint(
        self.min_repeat - 1, self.max_repeat)
    return self.curr_act

  def end_episode(self):
    self.curr_act = None
    self.repeats_left = 0