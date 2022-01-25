import copy

from gym import spaces
import numpy as np
import torch
from torch.nn import functional as F
import wandb

from .Rainbow import Agent
from .Rainbow import ReplayMemory as RainbowMemory
from .base import BaseAgent

class RainbowAgent(BaseAgent):
  def __init__(self, env, args, custom_encoder=None, repr_learner=None, tracked=False):
    self.args = args
    self.env = env
    self.repr_learner = repr_learner

    if torch.cuda.is_available() and not self.args.disable_cuda:
      self.args.device = torch.device('cuda')
      torch.cuda.manual_seed(np.random.randint(1, 10000))
      torch.backends.cudnn.enabled = self.args.enable_cudnn
    else:
      self.args.device = torch.device('cpu')
    
    # Change environment variable to match expected interface
    env = copy.copy(env)
    if type(env.action_space) != spaces.Discrete:
      raise Exception('Rainbow only supports discrete action spaces!')
    self.obs_dim = list(self.env.observation_space.shape)

    # Step seeds
    np.random.seed(self.args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
      
    # Instantiate model
    self.args.history_length = self.obs_dim[0]
    self.dqn = Agent(self.args, self.obs_dim,
      self.env.action_space.n, custom_encoder)

    if tracked:
      wandb.watch(self.dqn.online_net)
      wandb.watch(self.dqn.target_net)

  def start_task(self, n_steps):
    # Reset metrics and seeds
    self.metrics = {'steps': [], 'rewards': [], 'Qs': [],
               'best_avg_reward': -float('inf')}
    self.all_rewards = []
    self.ep_rewards = []

    self.mem = RainbowMemory(self.args, self.args.memory_capacity, self.obs_dim)
    # self.args.T_max = n_steps
    self.priority_weight_increase = (1 - self.args.priority_weight) \
      / (n_steps - self.args.learn_start)

    # Construct validation memory
    self.val_mem = RainbowMemory(self.args, self.args.evaluation_size, self.obs_dim)

    self.last_update_episode = 0
    self.step_idx = 1

  def train_representation(self):
    raise NotImplementedError('Representation learning not implemented with Rainbow!')
    # # TODO: This needs to be made into an unweighted sampling
    # _, obs, acts, returns, next_obs, _, nonterminals = self.mem.sample(self.repr_learner.batch_size)
    # acts = F.one_hot(acts, self.env.action_space.n)
    # dones = 1 - nonterminals
    # self.repr_learner.train([obs, acts, returns, next_obs, dones])

  def sample_act(self, obs):
    if self.step_idx % self.args.replay_frequency == 0:
      self.dqn.reset_noise() # Draw a new set of noisy weights

    if obs.device != self.args.device:
      obs = obs.to(self.args.device)
    action = self.dqn.act(obs) # Choose an action greedily (with noisy weights)
    return action

  def process_step_data(self, transition_data):
    obs, action, reward, _, done = transition_data
    self.ep_rewards.append(reward)
    # Clip rewards
    if self.args.reward_clip > 0:
      reward = max(min(reward, self.args.reward_clip), -self.args.reward_clip)
    self.mem.append(obs, action, reward, done) # Append transition to memory

  def end_episode(self):
    self.all_rewards.append(sum(self.ep_rewards))
    self.ep_rewards = []

  def end_step(self):
    # Train and test
    # if self.step_idx % 500 == 0:
    #   print('Step:', self.step_idx)
    if self.step_idx >= self.args.learn_start:
      # Anneal importance sampling weight β to 1
      self.mem.priority_weight = min(self.mem.priority_weight + \
                                      self.priority_weight_increase, 1)

      # Train with n-step distributional double-Q learning
      if self.step_idx % self.args.replay_frequency == 0:
        loss = self.dqn.learn(self.mem)
        wandb.log({'task_agent_loss': loss})

      # TODO: Move this logging to the simulation
      if self.step_idx % self.args.evaluation_interval == 0:
        print('Step: {}\t# Episodes: {}\tAvg ep reward: {:.2f}'.format(
            self.step_idx,
            len(self.all_rewards) - self.last_update_episode,
            np.mean(self.all_rewards[self.last_update_episode:])))
        self.last_update_episode = len(self.all_rewards)

      # Update target network
      if self.step_idx % self.args.target_update == 0:
        self.dqn.update_target_net()

      # Train representation network
      if self.repr_learner is not None and \
          self.step_idx % self.repr_learner.update_freq == 0:
        self.train_representation()

    self.step_idx += 1