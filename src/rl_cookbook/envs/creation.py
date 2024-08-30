import gym
from gym.wrappers import TimeLimit, TransformObservation
import gym_gridworld
import torch

from .wrappers import GridWorldWrapper, SimpleMapWrapper
from .wrappers import ATARI_WRAPPERS, PROCGEN_WRAPPERS, GYM_1D_WRAPPERS, N_FRAME_STACK


def create_gridworld_env(max_steps=500):
  """Creates a GridWorld environment with specified maximum steps.

  Args:
    max_steps: Maximum number of steps allowed in the environment.

  Returns:
    A wrapped GridWorld environment.
  """
  global N_FRAME_STACK
  N_FRAME_STACK = 1

  env = gym.make('gridworld-v0')
  env = GridWorldWrapper(env)
  env = TimeLimit(env, max_steps)
  env = TransformObservation(env, torch.FloatTensor)
  return env

def create_simple_gridworld_env(randomized=False, max_steps=500):
  """Creates a simple GridWorld environment.

  Args:
    randomized: Whether to use a randomized map.
    max_steps: Maximum number of steps allowed in the environment.

  Returns:
    A wrapped simple GridWorld environment.
  """
  env = create_gridworld_env(max_steps)
  env = SimpleMapWrapper(env, randomized)
  return env


### Atari Env ###


def create_atari_env(env_name):
  """Creates an Atari environment with standard wrappers.

  Args:
    env_name: Name of the Atari environment.

  Returns:
    A wrapped Atari environment.
  """
  global N_FRAME_STACK
  N_FRAME_STACK = 4

  env = gym.make(env_name)
  for wrapper in ATARI_WRAPPERS:
    env = wrapper(env)
  return env

def create_breakout_env():
  """Creates a Breakout environment."""
  return create_atari_env('BreakoutNoFrameskip-v4')

def create_crazy_climber_env():
  """Creates a Crazy Climber environment."""
  return create_atari_env('CrazyClimberNoFrameskip-v4')


### Gym Envs ###


def create_gym_1d_env(env_name):
  """Creates a 1D Gym environment with standard wrappers.

  Args:
    env_name: Name of the Gym environment.

  Returns:
    A wrapped 1D Gym environment.

  Raises:
    AssertionError: If the environment is not supported.
  """
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


### Procgen ### 


def create_procgen_env(env_name):
  """Creates a Procgen environment with standard wrappers.

  Args:
    env_name: Name of the Procgen environment.

  Returns:
    A wrapped Procgen environment.
  """
  global N_FRAME_STACK
  N_FRAME_STACK = 3

  env = gym.make(env_name)
  for wrapper in PROCGEN_WRAPPERS:
    env = wrapper(env)
  return env


### General ###


def make_env(env_name):
  """Creates an environment based on the given name.

  This function serves as a factory method for creating various types of
  environments including GridWorld, Gym 1D, Procgen, and Atari.

  Args:
    env_name: Name of the environment to create.

  Returns:
    An instance of the specified environment.
  """
  if 'gridworld' in env_name.lower():
    if 'random' in env_name.lower():
      return create_simple_gridworld_env(True)
    else:
      return create_simple_gridworld_env(False)
  elif env_name in SUPPORTED_GYM_1D_ENVS:
    return create_gym_1d_env(env_name)
  elif 'procgen' in env_name.lower():
    return create_procgen_env(env_name)
  else:
    return create_atari_env(env_name)
  

# if __name__ == '__main__':
#   # env = make_env('procgen:procgen-coinrun-v0')
#   env = make_env('procgen:procgen-caveflyer-v0')
#   print(env.observation_space)
#   obs = env.reset()
#   print(obs.shape, type(obs))
  
#   for i in range(3):
#     obs, r, _, _ = env.step(env.action_space.sample())
  
#   print(obs.shape, type(obs), obs.reshape(-1).min(), obs.reshape(-1).max(), r)