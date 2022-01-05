from .base import BaseAgent

from gym import spaces
import numpy as np


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