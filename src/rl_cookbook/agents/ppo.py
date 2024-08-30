from .base import BaseAgent, ExperienceBufferMixin
from ..envs.data import RewardNormalizer

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
import wandb


class PPOAgent(BaseAgent, ExperienceBufferMixin):
  """Proximal Policy Optimization (PPO) agent.

  This agent implements the PPO algorithm for reinforcement learning.

  Args:
    env: The environment the agent interacts with.
    policy: The policy network.
    critic: The value function network.
    calculate_rewards: Optional function to calculate custom rewards.
    batch_size: Number of samples per batch for training.
    update_freq: Number of steps between each policy update.
    log_freq: Frequency of logging training statistics.
    epsilon: Exploration rate for epsilon-greedy action selection.
    lr: Learning rate for the optimizer.
    gamma: Discount factor for future rewards.
    ppo_iters: Number of epochs to optimize the surrogate objective.
    ppo_clip: Clipping parameter for PPO.
    value_coef: Coefficient for the value function loss.
    entropy_coef: Coefficient for the entropy bonus.
    normalize_rewards: Whether to normalize rewards.
  """

  def __init__(self, env, policy, critic, calculate_rewards=None, batch_size=32,
               update_freq=128, log_freq=100, epsilon=0.05, lr=3e-4,
               gamma=0.99, ppo_iters=20, ppo_clip=0.2, value_coef=0.5,
               entropy_coef=0.003, normalize_rewards=True):
    super().__init__()

    assert batch_size <= update_freq, 'Batch size must be <= update freq!'

    self.n_acts = env.action_space.n
    self.policy = policy
    self.critic = critic
    self.calculate_rewards = calculate_rewards
    self.batch_size = batch_size
    self.update_freq = update_freq
    self.log_freq = log_freq
    self.device = next(self.policy.parameters()).device
    self.epsilon = epsilon
    self.gamma = gamma
    self.ppo_iters = ppo_iters
    self.ppo_clip = ppo_clip
    self.value_coef = value_coef
    self.entropy_coef = entropy_coef
    self.policy_losses = []
    self.critic_losses = []
    self.step_idx = 1

    if normalize_rewards:
      self.reward_normalizer = RewardNormalizer()
    else:
      self.reward_normalizer = None

    self.optimizer = optim.Adam(list(self.policy.parameters()) \
      + list(self.critic.parameters()), lr=lr)

  def process_step_data(self, transition_data):
    """Process a single step of experience data."""
    self.append_buffer(transition_data)
    self.step_idx += 1

  def sample_act(self, obs):
    """Sample an action given an observation.

    Args:
      obs: The current observation.

    Returns:
      An action sampled from the policy.
    """
    if np.random.rand() < self.epsilon:
      return np.random.randint(0, self.n_acts)

    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
    obs = obs.unsqueeze(0)
    with torch.no_grad():
      logits = self.policy(obs)
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    return np.random.choice(self.n_acts, p=probs)

  def prepare_recent_batch_data(self):
    """Prepare the most recent batch of data for training."""
    batch_data = self.get_buffer_recent_data(self.update_freq)
    self.clear_buffer()
    batch_data = [torch.tensor(e, dtype=torch.float32) \
       for e in batch_data]
    batch_data[1] = batch_data[1].long()
    return batch_data

  def train(self):
    """Perform a training update using the PPO algorithm."""
    self.policy.train()

    batch_data = self.prepare_recent_batch_data()
    _, _, rewards, next_obs, dones = batch_data
    if self.calculate_rewards is not None:
        rewards = self.calculate_rewards(batch_data).cpu()

    if self.reward_normalizer is not None:
      rewards = self.reward_normalizer.normalize(rewards)

    # Bootstrap rewards if episode is not done
    if not dones[-1]:
      dones[-1] = 1
      next_value = self.critic(next_obs[-1:].to(self.device)).squeeze()
      next_value = next_value.detach().cpu()
      rewards[-1] += self.gamma * next_value

    # Calculate returns
    # TODO: Use generalized advantage estimation
    returns = torch.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for i in range(len(rewards)-2, -1, -1):
      returns[i] = rewards[i] + self.gamma * returns[i+1] * (1 - dones[i])
      
    obs, acts, _, _, _ = [e.to(self.device) for e in batch_data]
    returns = returns.to(self.device)

    with torch.no_grad():
      values = self.critic(obs).squeeze(1)
      advantages = returns - values
      logits = self.policy(obs)
    probs = F.softmax(logits, dim=-1)
    old_act_probs = probs.gather(1, acts.unsqueeze(1)).squeeze(1)

    train_data = {
      'obs': obs,
      'act': acts,
      'old_act_probs': old_act_probs,
      'old_values': values,
      'advantages': advantages,
      'returns': returns}

    policy_losses = []
    critic_losses = []
    for _ in range(self.ppo_iters):
      zipped_buffer = list(zip(*train_data.values()))
      np.random.shuffle(zipped_buffer)
      train_buffer = list(zip(*zipped_buffer))
      train_data = {k: torch.stack(v) for k, v in \
        zip(train_data.keys(), train_buffer)}

      # Break the data into mini-batches for the model updates
      for batch_idx in range(int(np.ceil(len(train_data['obs']) / self.batch_size))):
        minibatch = {k: v[batch_idx * self.batch_size: \
          (batch_idx + 1) * self.batch_size] \
          for k, v in train_data.items()}
          
        # Calculate new action probabilities and values for the epoch
        new_values = self.critic(minibatch['obs'])
        new_act_probs = F.softmax(self.policy(minibatch['obs']), dim=-1)
        policy_entropy = Categorical(probs=new_act_probs).entropy()
        new_act_probs = new_act_probs.gather(1, minibatch['act'].unsqueeze(1))
        new_act_probs = new_act_probs.squeeze(1)
        new_values = new_values.squeeze(1)

        # Calulcate the value loss
        value_loss = F.mse_loss(new_values, minibatch['returns'])

        # Calculate the policy loss
        # print(new_act_probs, minibatch['old_act_probs'])
        policy_ratio = new_act_probs / (minibatch['old_act_probs'] + 1e-7)
        clipped_policy_ratio = torch.clamp(policy_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
        policy_loss = torch.min(policy_ratio * minibatch['advantages'],
          clipped_policy_ratio * minibatch['advantages'])
        policy_loss = -policy_loss.mean()
        entropy_loss = -policy_entropy.mean()

        total_loss = policy_loss + self.value_coef * value_loss \
          + self.entropy_coef * entropy_loss

        policy_losses.append(policy_loss.item())
        critic_losses.append(value_loss.item())

        # Update the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    self.policy_losses.append(np.mean(policy_losses))
    self.critic_losses.append(np.mean(critic_losses))
    wandb.log({'ppo_policy_loss': self.policy_losses[-1],
               'ppo_critic_loss': self.critic_losses[-1]})

    return np.mean(policy_losses) + np.mean(critic_losses)

  def end_step(self):
    """Perform end-of-step operations, including training and logging."""
    # Perform a training step
    if self.step_idx % self.update_freq == 0:
      self.train()

    # Log policy stats
    if self.log_freq > 0 and len(self.policy_losses) >= self.log_freq:
      print('Step: {} | Policy loss: {:.4f} | Critic loss: {:.4f}'.format(
          self.step_idx, np.mean(self.policy_losses), np.mean(self.critic_losses)))
      self.policy_losses = []
      self.critic_losses = []