import numpy as np
import wandb

from ..agents import BaseAgent
from ..envs.data import DiscreteEntropyTracker


def train_exploration_model(agent: BaseAgent, env, n_steps):
  act_entropy_tracker = DiscreteEntropyTracker(env.action_space.n)
  agent.start_task(n_steps)

  obs = env.reset()
  step_idx = 0
  episode_idx = 0
  ep_reward = 0
  wandb.log({'env_explore_episode': episode_idx})
  while step_idx < n_steps:
    # Take an action and process new data
    act = agent.sample_act(obs)
    next_obs, reward, done, _ = env.step(act)
    ep_reward += reward
    agent.process_step_data([obs, act, 0, next_obs, done])

    # Logging
    act_entropy = act_entropy_tracker.calc_entropy(act)
    wandb.log({
      'env_explore_step': step_idx,
      'env_explore_act_entropy': act_entropy,
      'env_explore_reward': reward})

    if done:
      agent.end_episode()
      obs = env.reset()
      episode_idx += 1
      wandb.log({'env_explore_episode': episode_idx})
      wandb.log({'env_explore_episode_reward': ep_reward})
      ep_reward = 0
    else:
      obs = next_obs

    agent.end_step()
    step_idx += 1

  agent.end_task()

def train_task_model(agent, env, n_steps, print_rewards=False, print_freq=5000):
  act_entropy_tracker = DiscreteEntropyTracker(env.action_space.n)
  agent.start_task(n_steps)

  obs = env.reset()
  step_idx = 0
  episode_idx = 0
  wandb.log({'env_task_episode': episode_idx})
  recent_rewards = []
  episode_rewards = []
  while step_idx < n_steps:
    # Take an action and process new data
    act = agent.sample_act(obs)
    next_obs, reward, done, _ = env.step(act)
    episode_rewards.append(reward)
    agent.process_step_data([obs, act, reward, next_obs, done])

    # Logging
    act_entropy = act_entropy_tracker.calc_entropy(act)
    wandb.log({
      'env_task_step': step_idx,
      'env_task_act_entropy': act_entropy,
      'env_task_reward': reward})

    if done:
      agent.end_episode()
      obs = env.reset()
      recent_rewards.append(np.sum(episode_rewards))
      wandb.log({'env_task_episode': episode_idx})
      wandb.log({'env_task_episode_reward': np.sum(episode_rewards)})
      episode_rewards = []
      episode_idx += 1
    else:
      obs = next_obs

    if print_rewards and step_idx > 0 and step_idx % print_freq == 0:
      print('Step: {} | Episodes: {} | Ep rewards: {:.4f}'.format(
        step_idx, len(recent_rewards), np.mean(recent_rewards)))
      recent_rewards = []

    agent.end_step()
    step_idx += 1

  agent.end_task()
