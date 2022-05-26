import copy

import cv2
import numpy as np
import gym
from gym.wrappers import AtariPreprocessing, FrameStack, TransformObservation
import torch


N_FRAME_STACK = 4

class NoRewardWrapper(gym.RewardWrapper):
  def __init__(self, env):
    super().__init__(env)

  def reward(self, _):
    return 0

class NoTerminationWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    obs, reward, _, info = self.env.step(action)
    return obs, reward, False, info

class RandomTerminationWrapper(gym.Wrapper):
  def __init__(self, env, termination_chance=0.01):
    super().__init__(env)
    self.termination_chance = termination_chance

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if np.random.random() < self.termination_chance:
      done = True
    return obs, reward, done, info

class GridWorldWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

  def observation(self, observation):
    # Convert to grayscale
    observation = np.sum(observation * np.array([[[0.2989, 0.5870, 0.1140]]]), axis=2)
    self.formatted_obs = observation
    return observation

  def jump_to_state(self, state):
      obs, reward, done, info = self.env.jump_to_state(state)
      return self.observation(obs), reward, done, info

class SimpleMapWrapper(gym.ObservationWrapper):
  def __init__(self, env, randomized=False, map_shape=(16, 16)):
    super().__init__(env)
    self.random_start = randomized
    self.env.unwrapped.obs_shape = map_shape + (3,)
    self.observation_space = gym.spaces.Box(
        low=0, high=1, shape=[1]+list(map_shape), dtype=np.float32)

    self._reset_start_map()

  def _reset_start_map(self):
    map = np.ones(self.observation_space.shape[1:], dtype=np.int64)
    map [1:-1, 1:-1] = 0

    if self.random_start:
      max_x, max_y = np.array(self.observation_space.shape[1:]) - 1
      map[np.random.randint(1, max_x), np.random.randint(1, max_y)] = 3
      new_pos = (np.random.randint(1, max_x), np.random.randint(1, max_y))
      while map[new_pos[0], new_pos[1]] == 3:
        new_pos = (np.random.randint(1, max_x), np.random.randint(1, max_y))
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

  def observation(self, observation):
    # observation = cv2.resize(observation, self.observation_space.shape[1:],
    #                          interpolation=cv2.INTER_AREA)
    observation = np.expand_dims(observation, 0)
    self.formatted_obs = observation
    return observation

  def jump_to_state(self, state):
      obs, reward, done, info = self.env.jump_to_state(state)
      return self.observation(obs), reward, done, info

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

class Custom2DWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        frame_skip=1,
        rescale_size=None,
        grayscale_obs=False,
        grayscale_newaxis=True,
        scale_obs=True):
      super().__init__(env)
      assert frame_skip > 0

      self.frame_skip = frame_skip
      self.rescale = rescale_size is not None 
      self.rescale_size = rescale_size
      self.grayscale_obs = grayscale_obs
      self.grayscale_newaxis = grayscale_newaxis
      self.scale_obs = scale_obs

      _low, _high, _obs_dtype = (
          (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
      )
      
      if self.rescale:
        _shape = (rescale_size, rescale_size, 1 if grayscale_obs else 3)
      else:
        _shape = (*self.observation_space.shape[:2], 1 if grayscale_obs else 3)
        
      if grayscale_obs and not grayscale_newaxis:
          _shape = _shape[:-1]  # Remove channel axis
      self.observation_space = gym.spaces.Box(
          low=_low, high=_high, shape=_shape, dtype=_obs_dtype
      )

    def step(self, action):
      R = 0.0
      for _ in range(self.frame_skip):
        obs, reward, done, info = self.env.step(action)
        R += reward
        self.game_over = done
        if done:
          break
      return self._preprocess_obs(obs), R, done, info

    def reset(self, **kwargs):
      # NoopReset
      obs = self.env.reset(**kwargs)
      return self._preprocess_obs(obs)

    def _preprocess_obs(self, obs):
      # Resize the dimensions
      if self.rescale:
        obs = cv2.resize(obs, (self.rescale_size, self.rescale_size),
                         interpolation=cv2.INTER_AREA)
        
      # Convert to grayscale
      if self.grayscale_obs:
        obs = np.sum(obs * np.array([[[0.2989, 0.5870, 0.1140]]]), axis=2,
                     keepdims=self.grayscale_newaxis)
      obs = obs.transpose(2, 0, 1)

      # Rescale obs to [0, 1]
      if self.scale_obs:
        obs = obs / 255.0
      
      return obs


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

PROCGEN_WRAPPERS = [
  lambda env: Custom2DWrapper(env),
  lambda env: FrameStack(env, N_FRAME_STACK),
  lambda env: TransformObservation(env, torch.FloatTensor),
  lambda env: TransformObservation(env, lambda x: x.view(-1, *x.shape[2:]))
]