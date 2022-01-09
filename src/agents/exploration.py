from torch._C import Value
from .base import BaseAgent, BaseRepresentationLearner, ExperienceBufferMixin

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
  def __init__(self, env, policy, repr_learner, batch_size=32,
               update_freq=16, log_freq=100, epsilon=0.05, lr=3e-4):
    ReprLearningMixin.__init__(self, env, repr_learner)
    super().__init__()

    self.optim_type = 'SUPERVISED' # 'REINFORCE'

    self.policy = policy
    self.batch_size = batch_size
    self.update_freq = update_freq
    self.log_freq = log_freq
    self.policy_device = next(self.policy.parameters()).device
    self.epsilon = epsilon
    self.policy_losses = []
    self.step_idx = 1

    self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

  def process_step_data(self, transition_data):
    self.process_repr_step_data(transition_data)
    self.step_idx += 1

  def sample_act(self, obs):
    if np.random.rand() < self.epsilon:
      return np.random.randint(0, self.n_acts)

    obs = torch.tensor(obs, dtype=torch.float32, device=self.policy_device)
    obs = obs.unsqueeze(0)
    with torch.no_grad():
      logits = self.policy(obs)
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    return np.random.choice(self.n_acts, p=probs)

  def train_policy(self):
    replace = self.buffer_size() < self.batch_size
    batch_data = self.sample_buffer(self.batch_size, replace)
    obs, act_idxs, rewards, next_obs, dones = batch_data
    acts = F.one_hot(act_idxs.long(), self.n_acts)

    obs, acts, _, next_obs, _ = \
      [torch.tensor(e, dtype=torch.float32).to(self.policy_device) \
        for e in (obs, acts, rewards, next_obs, dones)]
    
    self.policy.train()

    logits = self.policy(obs)
    act_idxs = torch.tensor(act_idxs, dtype=torch.long).to(self.policy_device)

    with torch.no_grad():
      repr_losses = self.repr_learner.calculate_losses(batch_data)

    if self.optim_type == 'SUPERVISED':
      act_logits = logits.gather(1, act_idxs.unsqueeze(1)).squeeze(1)
      losses = (repr_losses - act_logits) ** 2
    elif self.optim_type == 'REINFORCE':
      rewards = repr_losses
      probs = F.softmax(logits, dim=-1)
      act_probs = probs.gather(1, act_idxs.unsqueeze(1))
      losses = -torch.log(act_probs) * rewards
    else:
      raise ValueError('Unknown optimization type!')

    # print(act_logits.shape, repr_losses.shape, 'SHOULD BE THE SAME SHAPE')
      
    loss = losses.mean()
    self.policy_losses.append(loss.item())

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

  def end_step(self):
    # Perform a training step
    if self.step_idx % self.update_freq == 0:
      self.train_policy()

    # Log policy stats
    if len(self.policy_losses) >= self.log_freq:
      print('Step: {} | Policy loss: {:.4f}'.format(
          self.step_idx, np.mean(self.policy_losses)))
      self.policy_losses = []