from .base import BaseAgent, BaseRepresentationLearner, ExperienceBufferMixin
from .ppo import PPOAgent

from gym import spaces
from einops import rearrange
import numpy as np
import torch
from torch.nn import functional as F
import wandb


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
    batch_data = self.sample_buffer(self.repr_learner.batch_size, replace)
    loss = self.repr_learner.train(batch_data)
    self.repr_losses.append(loss)

    wandb.log({'repr_loss': loss})
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
    wandb.log({'explorer_reward': repr_losses.mean().item()})
    return repr_losses


class MaxEntropyExplorerAgent(PPOAgent, ReprLearningMixin):
  def __init__(self, env, policy, critic, repr_learner, knn_k=5,
               device=torch.device('cuda'), **kwargs):
    ReprLearningMixin.__init__(self, env, repr_learner)
    PPOAgent.__init__(self, env, policy, critic,
      calculate_rewards=self.calculate_ppo_rewards,
      **kwargs)

    self.knn_k = knn_k
    self.device = device

  def process_step_data(self, transition_data):
    PPOAgent.process_step_data(self, transition_data)
    self.process_repr_step()

  def calculate_ppo_rewards(self, batch_data):
    obs = batch_data[0]
    repr_device = next(self.repr_learner.encoder.parameters()).device
    with torch.no_grad():
      latent_states = self.repr_learner.encoder(obs.to(repr_device))
    if repr_device != self.device:
      latent_states = latent_states.to(self.device)

    # Source: https://github.com/rll-research/url_benchmark/blob/bb98f0c6d78b3c467fb5a9fa5bbba3b7c0250397/utils.py#L289
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    # TODO: Make sure states aren't getting compared to themselves and duplicate pairs
    sim_matrix = torch.norm(
      rearrange(latent_states, 'b d -> b 1 d') -
      rearrange(latent_states, 'b d -> 1 b d'),
      dim=-1, p=2)
    neighbor_diffs, _ = sim_matrix.topk(
      self.knn_k, dim=-1, largest=False, sorted=False) # (b, k)
    # TODO: Log closest neighbor distance
    rewards = torch.log(1 + neighbor_diffs.mean(dim=-1))
    rewards = rewards.cpu()
    wandb.log({'explorer_reward': rewards.mean().item()})

    return rewards