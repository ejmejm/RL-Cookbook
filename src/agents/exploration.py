from .base import BaseAgent, ExperienceBufferMixin

from gym import spaces
import numpy as np
from torch.nn import functional as F


class EzExplorerAgent(BaseAgent, ExperienceBufferMixin):
  def __init__(self, env, min_repeat=1, max_repeat=6, repr_learner=None):
    if type(env.action_space) != spaces.Discrete:
      raise Exception('EzExplorerAgent only supports discrete action spaces!')
    super().__init__()

    self.n_acts = env.action_space.n
    self.min_repeat = min_repeat
    self.max_repeat = max_repeat
    self.repr_learner = repr_learner
    self.repr_losses = []

    self.curr_act = None
    self.repeats_left = 0
    self.step_idx = 0

  def train_representation(self):
    replace = self.step_idx < self.buffer_size()
    obs, acts, rewards, next_obs, dones = \
      self.sample_buffer(self.repr_learner.batch_size, replace)
    acts = F.one_hot(acts.long(), self.n_acts)
    loss = self.repr_learner.train([obs, acts, rewards, next_obs, dones])
    self.repr_losses.append(loss)

    if len(self.repr_losses) >= self.repr_learner.log_freq:
      print('Step #{} | Repr loss: {:.4f}'.format(self.step_idx, np.mean(self.repr_losses)))
      self.repr_losses = []

  def process_step_data(self, transition_data):
    if self.repr_learner is not None:
      self.append_buffer(transition_data)

      if (self.step_idx + 1) % self.repr_learner.update_freq == 0:
        self.train_representation()

    self.step_idx += 1

  def sample_act(self, _):
    if self.repeats_left > 0:
      self.repeats_left -= 1
      return self.curr_act
    
    self.curr_act = np.random.randint(0, self.n_acts)
    self.repeats_left = np.random.randint(
        self.min_repeat - 1, self.max_repeat)
    return self.curr_act

  def end_episode(self):
    self.curr_act = None
    self.repeats_left = 0