import copy

from .base import BaseAgent, ExperienceBufferMixin
from ..envs.data import RewardNormalizer

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
import wandb


class DDDQNAgent(BaseAgent, ExperienceBufferMixin):
  def __init__(self, env, model, calculate_rewards=None, batch_size=128,
               update_freq=1, log_freq=100, lr=3e-4, epsilon=0.05,
               gamma=0.99, n_step=12, target_update_freq=200, learning_start=1600,
               normalize_rewards=False):
    super().__init__()

    self.n_acts = env.action_space.n
    self.model = model
    self.calculate_rewards = calculate_rewards
    self.batch_size = batch_size
    self.update_freq = update_freq
    self.log_freq = log_freq
    self.device = next(self.model.parameters()).device
    self.epsilon = epsilon
    self.learning_start = learning_start
    self.gamma = gamma
    self.n_step = n_step
    self.target_update_freq = target_update_freq # In number of updates
    self.losses = []
    self.step_idx = 1
    # Stored transitions, waiting for n_steps before adding
    # to buffer to calculate n_step rewards
    self.n_step_buffer = []
    self._update_target_network()

    if calculate_rewards is not None:
      raise ValueError('calculate rewards is not yet supported for DDDQN')

    if normalize_rewards:
      self.reward_normalizer = RewardNormalizer()
    else:
      self.reward_normalizer = None

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

  def _update_target_network(self):
      self.target_model = copy.deepcopy(self.model)

  def _gen_buffer_entry(self):
    # Input transition data format: [obs, act, reward, next_obs, done]
    n_step_reward = 0
    for i in range(len(self.n_step_buffer)):
        reward = self.n_step_buffer[i][2]
        n_step_reward += reward * self.gamma ** i
    next_gamma = self.gamma ** (i + 1)
    start_trans, end_trans = self.n_step_buffer[0], self.n_step_buffer[-1]

    # Output transition data format: [obs, act, n_step_reward, final_obs, done, final_gamma]
    buffer_entry = [start_trans[0], start_trans[1], n_step_reward,
                    end_trans[3], end_trans[4], next_gamma]

    return buffer_entry

  def process_step_data(self, transition_data):
    # Normalize the reward, then put it into the n_step_buffer
    if self.reward_normalizer is not None:
      # TODO: Speed up reward normalization, when done every step this
      # will probably take a noticeably amount of compute
      # TODO: Reward normalization currently doesn't work well for this,
      # need to figure out why and if theres a bug
      transition_data[2] = self.reward_normalizer.normalize([transition_data[2]])[0]
    
    self.n_step_buffer.append(transition_data)
    self.n_step_buffer = self.n_step_buffer[-self.n_step:]
    if len(self.n_step_buffer) < self.n_step:
        return

    new_transition = self._gen_buffer_entry()
    self.append_buffer(new_transition)

  def end_episode(self):
    while len(self.n_step_buffer) > 1:
      self.n_step_buffer = self.n_step_buffer[1:]
      new_transition = self._gen_buffer_entry()
      self.append_buffer(new_transition)
    self.n_step_buffer = []

  def sample_act(self, obs):
    if np.random.rand() < self.epsilon:
      return np.random.randint(0, self.n_acts)

    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
    obs = obs.unsqueeze(0)
    with torch.no_grad():
      q_vals = self.model(obs).cpu().numpy()[0]
    if (q_vals == q_vals[0]).all():
      return np.random.randint(0, self.n_acts)
    return np.argmax(q_vals)

  def prepare_batch_data(self):
    if len(self.exp_buffer) < self.batch_size:
        replace = True
    else:
        replace = False
    batch_data = self.sample_buffer(self.batch_size, replace=replace)
    batch_data = [torch.tensor(e, dtype=torch.float32, device=self.device) \
       for e in batch_data]
    batch_data[1] = batch_data[1].long()
    batch_data[4] = batch_data[4].int()
    return batch_data

  def calculate_losses(self):
    batch_data = self.prepare_batch_data()
    obs, acts, n_step_rewards, final_obs, terminals, final_gammas = batch_data
    final_q_values = self.model(final_obs)
    max_next_acts = torch.argmax(final_q_values, dim=1, keepdim=True).detach()
    
    final_q_values = self.target_model(final_obs)
    max_final_q_values = final_q_values.gather(index=max_next_acts, dim=1)
    max_final_q_values = max_final_q_values.view(-1).detach()
    terminal_mods = 1 - terminals
    target_qs = n_step_rewards + terminal_mods * final_gammas * max_final_q_values

    pred_qs = self.model(obs)
    pred_qs = pred_qs.gather(index=acts.view(-1, 1), dim=1).view(-1)
    
    losses = (target_qs.detach() - pred_qs) ** 2
    return losses

  def train(self):
    self.model.train()

    losses = self.calculate_losses()
    loss = losses.mean()
    wandb.log({'dqn_loss': loss.item()})
    self.losses.append(loss.item())
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

  def end_step(self):
    # Perform a training step
    if self.step_idx >= self.learning_start and \
       self.step_idx % self.update_freq == 0:
      self.train()

    # Update the target network
    if self.step_idx % self.target_update_freq == 0:
        self._update_target_network()

    # Log loss stats
    if self.log_freq > 0 and len(self.losses) >= self.log_freq:
      print('Step: {} | DDDQN loss: {:.4f}'.format(
          self.step_idx, np.mean(self.losses)))
      self.losses = []

    self.step_idx += 1