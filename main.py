from abc import ABC, abstractmethod
import os
import sys
import copy
import types
import gym
from gym.wrappers import AtariPreprocessing, FrameStack, TransformObservation
from gym.wrappers import TimeLimit
import gym_gridworld
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import einops
import seaborn as sns
import cv2
from tqdm.notebook import tqdm

from src import Agent, DEFAULT_RAINBOW_ARGS
from src import ReplayMemory as RainbowMemory
from src import NoisyLinear
# from src.test import test as test_rainbow

sns.set()

### Gridworld Env ###

class GridWorldWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

    obs_shape = (1, 16, 16) # (1, 16, 16)
    self.observation_space = gym.spaces.Box(
        low=0, high=1, shape=obs_shape, dtype=np.float32)

  def observation(self, observation):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, self.observation_space.shape[1:],
                             interpolation=cv2.INTER_AREA)
    observation = np.expand_dims(observation, 0)
    return observation

class SimpleMapWrapper(gym.Wrapper):
  def __init__(self, env, randomized=False):
    super().__init__(env)
    self.random_start = randomized
    self._reset_start_map()

  def _reset_start_map(self):
    map = np.ones((16, 16), dtype=np.int64)
    map [1:-1, 1:-1] = 0

    get_random_pos = lambda: np.random.randint(1, 15)
    if self.random_start:
      map[get_random_pos(), get_random_pos()] = 3
      new_pos = (get_random_pos(), get_random_pos())
      while map[new_pos[0], new_pos[1]] == 3:
        new_pos = (get_random_pos(), get_random_pos())
      map[new_pos[0], new_pos[1]] = 4
    else:
      map[-3, 2] = 3
      map[-6, 4] = 4

    uenv = self.unwrapped
    uenv.start_grid_map = map
    uenv.current_grid_map = copy.deepcopy(uenv.start_grid_map)  # current grid map
    uenv.observation = uenv._gridmap_to_observation(uenv.start_grid_map)
    uenv.grid_map_shape = uenv.start_grid_map.shape

    uenv.agent_start_state, uenv.agent_target_state = \
      uenv._get_agent_start_target_state(uenv.start_grid_map)
    uenv.agent_state = copy.deepcopy(uenv.agent_start_state)

  def reset(self):
    if self.random_start:
      self._reset_start_map()
    return super().reset()

def create_gridworld_env(max_steps=1000):
  global N_FRAME_STACK
  N_FRAME_STACK = 1

  env = gym.make('gridworld-v0')
  env = GridWorldWrapper(env)
  env = TimeLimit(env, max_steps)
  env = TransformObservation(env, torch.FloatTensor)
  return env

def create_simple_gridworld_env(randomized=False, max_steps=1000):
  env = create_gridworld_env(max_steps)
  env = SimpleMapWrapper(env, randomized)
  return env


### Atari Env ###

atari_wrappers = [
  lambda env: AtariPreprocessing(env, scale_obs=True),
  lambda env: FrameStack(env, N_FRAME_STACK),
  lambda env: TransformObservation(env, torch.FloatTensor)
]

def create_atari_env(env_name):
  global N_FRAME_STACK
  N_FRAME_STACK = 4

  env = gym.make(env_name)
  for wrapper in atari_wrappers:
    env = wrapper(env)
  return env

def create_breakout_env():
  return create_atari_env('BreakoutNoFrameskip-v4')

def create_crazy_climber_env():
  return create_atari_env('CrazyClimberNoFrameskip-v4')

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
    if type(env.action_space) != gym.spaces.Discrete:
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


# class FuturePredictionAgent(EzExplorerAgent, )


def create_small_convs(input_dim):
  return nn.Sequential(
        nn.Conv2d(input_dim, 8, 4, 2),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1),
        nn.ReLU()
    )
    
class RainbowAgent(BaseAgent):
  def __init__(self, args, env, custom_encoder=None):
    self.args = args
    if torch.cuda.is_available() and not self.args.disable_cuda:
      self.args.device = torch.device('cuda')
      torch.cuda.manual_seed(np.random.randint(1, 10000))
      torch.backends.cudnn.enabled = self.args.enable_cudnn
    else:
      self.args.device = torch.device('cpu')
    
    # Change environment variable to match expected interface
    env = copy.copy(env)
    if type(env.action_space) != gym.spaces.Discrete:
      raise Exception('Rainbow only supports discrete action spaces!')
    self.env = env
    self.obs_dim = list(self.env.observation_space.shape)

    # Step seeds
    np.random.seed(self.args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
      
    # Instantiate model
    self.args.history_length = self.obs_dim[0]
    self.dqn = Agent(self.args, self.obs_dim,
      self.env.action_space.n, custom_encoder)

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

  def sample_act(self, obs):
    if self.step_idx % self.args.replay_frequency == 0:
      self.dqn.reset_noise()  # Draw a new set of noisy weights

    if obs.device != self.args.device:
      obs = obs.to(self.args.device)
    action = self.dqn.act(obs)  # Choose an action greedily (with noisy weights)
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
        self.dqn.learn(self.mem)

      if self.step_idx % self.args.evaluation_interval == 0:
        print('Step: {}\t# Episodes: {}\tAvg ep reward: {:.2f}'.format(
            self.step_idx,
            len(self.all_rewards) - self.last_update_episode,
            np.mean(self.all_rewards[self.last_update_episode:])))
        self.last_update_episode = len(self.all_rewards)

      # Update target network
      if self.step_idx % self.args.target_update == 0:
        self.dqn.update_target_net()

    self.step_idx += 1

env = create_simple_gridworld_env()
# env = create_breakout_env()
custom_encoder = None
if env.observation_space.shape[1] <= 42:
  custom_encoder = create_small_convs(N_FRAME_STACK)
agent = RainbowAgent(DEFAULT_RAINBOW_ARGS, env, custom_encoder)

agent.start_task(1000)
obs = env.reset()
act = agent.sample_act(obs)
print('Act:', act)
obs, reward, done, _ = env.step(act)
agent.process_step_data((obs, act, reward, obs, done))
agent.end_step()