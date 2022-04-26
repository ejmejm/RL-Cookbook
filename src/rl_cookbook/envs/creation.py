import gym
from gym.wrappers import TimeLimit, TransformObservation
import gym_gridworld
import torch

from .wrappers import GridWorldWrapper, SimpleMapWrapper
from .wrappers import ATARI_WRAPPERS, GYM_1D_WRAPPERS, N_FRAME_STACK


def create_gridworld_env(max_steps=500):
  global N_FRAME_STACK
  N_FRAME_STACK = 1

  env = gym.make('gridworld-v0')
  env = GridWorldWrapper(env)
  env = TimeLimit(env, max_steps)
  env = TransformObservation(env, torch.FloatTensor)
  return env

def create_simple_gridworld_env(randomized=False, max_steps=500):
  env = create_gridworld_env(max_steps)
  env = SimpleMapWrapper(env, randomized)
  return env


### Atari Env ###


def create_atari_env(env_name):
  global N_FRAME_STACK
  N_FRAME_STACK = 4

  env = gym.make(env_name)
  for wrapper in ATARI_WRAPPERS:
    env = wrapper(env)
  return env

def create_breakout_env():
  return create_atari_env('BreakoutNoFrameskip-v4')

def create_crazy_climber_env():
  return create_atari_env('CrazyClimberNoFrameskip-v4')


### Gym Envs ###


def create_gym_1d_env(env_name):
  assert env_name in SUPPORTED_GYM_1D_ENVS, \
    f'Unsupported gym 1d env: {env_name}'

  global N_FRAME_STACK
  N_FRAME_STACK = 4

  env = gym.make(env_name)
  for wrapper in GYM_1D_WRAPPERS:
    env = wrapper(env)
  return env

SUPPORTED_GYM_1D_ENVS = set([
  'MountainCar-v0',
  'Acrobot-v1'
])


### General ###


def make_env(env_name):
  if 'gridworld' in env_name.lower():
    if 'random' in env_name.lower():
      return create_simple_gridworld_env(True)
    else:
      return create_simple_gridworld_env(False)
  elif env_name in SUPPORTED_GYM_1D_ENVS:
    return create_gym_1d_env(env_name)
  else:
    return create_atari_env(env_name)