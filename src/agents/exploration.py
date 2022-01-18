from .base import BaseAgent, BaseRepresentationLearner, ExperienceBufferMixin
from .ppo import PPOAgent
from ..envs.data import RewardNormalizer

from gym import spaces
import numpy as np
import torch
from torch.nn import functional as F


class ReprLearningMixin():
  def __init__(self, env, repr_learner: BaseRepresentationLearner):
    if type(env.action_space) != spaces.Discrete:
      raise Exception('ReprLearningMixin only supports discrete action spaces!')

    self.n_acts = env.action_space.n
    self.repr_learner = repr_learner
    self.repr_losses = []

    self.repr_step_idx = 1

  def train_representation(self):
    replace = (self.repr_step_idx - 1) < self.buffer_size()
    obs, acts, rewards, next_obs, dones = \
      self.sample_buffer(self.repr_learner.batch_size, replace)
    acts = F.one_hot(acts.long(), self.n_acts)
    loss = self.repr_learner.train([obs, acts, rewards, next_obs, dones])
    self.repr_losses.append(loss)

    if len(self.repr_losses) >= self.repr_learner.log_freq:
      print('Step: {} | Repr loss: {:.4f}'.format(self.repr_step_idx, np.mean(self.repr_losses)))
      self.repr_losses = []

  def process_repr_step(self, transition_data=None):
    if transition_data is not None:
      self.append_buffer(transition_data)

    if self.repr_step_idx % self.repr_learner.update_freq == 0:
      self.train_representation()

    self.repr_step_idx += 1


class EzExplorerAgent(BaseAgent, ExperienceBufferMixin, ReprLearningMixin):
  def __init__(self, env, min_repeat=1, max_repeat=6, repr_learner=None):
    super().__init__()
    self.enable_repr_learning = repr_learner is not None
    if self.enable_repr_learning:
      ReprLearningMixin.__init__(self, env, repr_learner)

    self.n_acts = env.action_space.n
    self.min_repeat = min_repeat
    self.max_repeat = max_repeat

    self.curr_act = None
    self.repeats_left = 0

  def process_step_data(self, transition_data):
    if self.enable_repr_learning:
      self.process_repr_step(transition_data)

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


class SurprisalExplorerAgent(PPOAgent, ReprLearningMixin):
  def __init__(self, env, policy, critic, repr_learner, **kwargs):
    ReprLearningMixin.__init__(self, env, repr_learner)
    PPOAgent.__init__(self, env, policy, critic,
      calculate_rewards=self.calculate_ppo_rewards,
      **kwargs)

  def process_step_data(self, transition_data):
    PPOAgent.process_step_data(self, transition_data)
    self.process_repr_step()

  def calculate_ppo_rewards(self, batch_data):
    with torch.no_grad():
      repr_losses = self.repr_learner.calculate_losses(batch_data)
    return repr_losses