from .base import BaseAgent, BaseRepresentationLearner, ExperienceBufferMixin
from ..envs.data import RewardNormalizer

from gym import spaces
import numpy as np
import torch
from torch import optim
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

  def process_repr_step_data(self, transition_data):
    if self.repr_learner is not None:
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
      self.process_repr_step_data(transition_data)

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


class SurprisalExplorerAgent(BaseAgent, ExperienceBufferMixin, ReprLearningMixin):
  def __init__(self, env, policy, critic, repr_learner, batch_size=32,
               update_freq=128, log_freq=100, epsilon=0.05, lr=3e-4,
               gamma=0.99, ppo_iters=20, ppo_clip=0.2, normalize_rewards=True):
    ReprLearningMixin.__init__(self, env, repr_learner)
    super().__init__()

    assert batch_size <= update_freq, 'Batch size must be <= update freq!'

    self.policy = policy
    self.critic = critic
    self.batch_size = batch_size
    self.update_freq = update_freq
    self.log_freq = log_freq
    self.device = next(self.policy.parameters()).device
    self.epsilon = epsilon
    self.gamma = gamma
    self.ppo_iters = ppo_iters
    self.ppo_clip = ppo_clip
    self.policy_losses = []
    self.critic_losses = []
    self.extrinsic_rewards = []
    self.intrinsic_rewards = []
    self.step_idx = 1

    if normalize_rewards:
      self.reward_normalizer = RewardNormalizer()
    else:
      self.reward_normalizer = None

    self.optimizer = optim.Adam(list(self.policy.parameters()) \
      + list(self.critic.parameters()), lr=lr)

  def process_step_data(self, transition_data):
    self.process_repr_step_data(transition_data)
    self.step_idx += 1

  def sample_act(self, obs):
    if np.random.rand() < self.epsilon:
      return np.random.randint(0, self.n_acts)

    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
    obs = obs.unsqueeze(0)
    with torch.no_grad():
      logits = self.policy(obs)
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    return np.random.choice(self.n_acts, p=probs)

  def prepare_recent_batch_data(self):
    batch_data = self.get_buffer_recent_data(self.update_freq)
    batch_data = [torch.tensor(e, dtype=torch.float32) \
       for e in batch_data]
    batch_data[1] = batch_data[1].long()
    return batch_data

  def train(self):
    self.policy.train()

    batch_data = self.prepare_recent_batch_data()
    _, _, extrinsic_rewards, next_obs, dones = batch_data

    # Calculate intrinsic rewards and returns
    with torch.no_grad():
      repr_losses = self.repr_learner.calculate_losses(batch_data)
    intrinsic_rewards = repr_losses
    # intrinsic_rewards = extrinsic_rewards
    if self.reward_normalizer is not None:
      intrinsic_rewards = self.reward_normalizer.normalize(intrinsic_rewards)

    # Bootstrap rewards if episode is not done
    if not dones[-1]:
      dones[-1] = 1
      next_value = self.critic(next_obs[-1:].to(self.device)).squeeze()
      next_value = next_value.detach().cpu()
      intrinsic_rewards[-1] += self.gamma * next_value

    # Calculate returns
    returns = torch.zeros(len(intrinsic_rewards))
    returns[-1] = intrinsic_rewards[-1]
    for i in range(len(intrinsic_rewards)-2, -1, -1):
      returns[i] = intrinsic_rewards[i] + self.gamma * returns[i+1] * (1 - dones[i])
      
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
        new_act_probs = new_act_probs.gather(1, minibatch['act'].unsqueeze(1))
        new_act_probs = new_act_probs.squeeze(1)
        new_values = new_values.squeeze(1)

        # Calulcate the value loss
        value_loss = F.mse_loss(new_values, minibatch['returns'])

        # Calculate the policy loss
        policy_ratio = new_act_probs / (minibatch['old_act_probs'] + 1e-7)
        clipped_policy_ratio = torch.clamp(policy_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
        policy_loss = torch.min(policy_ratio * minibatch['advantages'],
          clipped_policy_ratio * minibatch['advantages'])
        policy_loss = -policy_loss.mean()

        total_loss = policy_loss + value_loss

        policy_losses.append(total_loss.item())
        critic_losses.append(value_loss.item())

        # Update the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    self.policy_losses.append(np.mean(policy_losses))
    self.critic_losses.append(np.mean(critic_losses))

    return np.mean(policy_losses) + np.mean(critic_losses)

  def end_step(self):
    # Perform a training step
    if self.step_idx % self.update_freq == 0:
      self.train()

    # Log policy stats
    if len(self.policy_losses) >= self.log_freq:
      print('Step: {} | Policy loss: {:.4f} | Critic loss: {:.4f}'.format(
          self.step_idx, np.mean(self.policy_losses), np.mean(self.critic_losses)))
      self.policy_losses = []
      self.critic_losses = []