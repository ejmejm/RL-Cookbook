import copy

import cv2
import numpy as np
import gym
from gym.wrappers import AtariPreprocessing, FrameStack, TransformObservation
import torch


N_FRAME_STACK = 4

class GridWorldWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

    obs_shape = (1, 16, 16)
    self.observation_space = gym.spaces.Box(
        low=0, high=1, shape=obs_shape, dtype=np.float32)

  def observation(self, observation):
    # Convert to grayscale
    observation = np.sum(observation * np.array([[[0.2989, 0.5870, 0.1140]]]), axis=2)
    observation = cv2.resize(observation, self.observation_space.shape[1:],
                             interpolation=cv2.INTER_AREA)
    observation = np.expand_dims(observation, 0)
    self.formatted_obs = observation
    return observation

  def jump_to_state(self, state):
      obs, reward, done, info = self.env.jump_to_state(state)
      return self.observation(obs), reward, done, info

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

class Scale1DObsWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

    obs_shape = env.observation_space.shape
    self.observation_space = gym.spaces.Box(
        low=np.zeros(obs_shape), high=np.ones(obs_shape),
        shape=env.observation_space.shape, dtype=np.float32)

  # Scales observations to [0, 1]
  def observation(self, observation):
    observation = observation.astype(np.float32)
    obs_space = self.observation_space
    obs_range = obs_space.high - obs_space.low
    observation = (observation - obs_space.low) / obs_range
    return observation

ATARI_WRAPPERS = [
  lambda env: AtariPreprocessing(env, scale_obs=True),
  lambda env: FrameStack(env, N_FRAME_STACK),
  lambda env: TransformObservation(env, torch.FloatTensor)
]

GYM_1D_WRAPPERS = [
  lambda env: Scale1DObsWrapper(env),
  lambda env: FrameStack(env, N_FRAME_STACK),
  lambda env: TransformObservation(env, torch.FloatTensor)
]