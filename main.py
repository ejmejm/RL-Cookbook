from torch import nn

from .agents import RainbowAgent
from .agents.Rainbow import DEFAULT_RAINBOW_ARGS
from .envs import *
from .training import *

def create_small_convs(input_dim):
  return nn.Sequential(
        nn.Conv2d(input_dim, 8, 4, 2),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1),
        nn.ReLU()
    )

env = create_simple_gridworld_env(True, 5000)
# env = create_breakout_env()
# env = create_crazy_climber_env()

custom_encoder = None
if env.observation_space.shape[1] <= 42:
  custom_encoder = create_small_convs(env.observation_space.shape[0])

# agent.start_task(1000)
# obs = env.reset()
# act = agent.sample_act(obs)
# print('Act:', act)
# obs, reward, done, _ = env.step(act)
# agent.process_step_data((obs, act, reward, obs, done))
# agent.end_step()

agent = RainbowAgent(DEFAULT_RAINBOW_ARGS, env, custom_encoder)
train_task_model(agent, env, int(1e5))