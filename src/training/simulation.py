import numpy as np

from ..agents import BaseAgent

def train_exploration_model(agent: BaseAgent, env, n_steps):
  agent.start_task(n_steps)

  obs = env.reset()
  step_idx = 0
  while step_idx < n_steps:
    act = agent.sample_act(obs)
    next_obs, _, done, _ = env.step(act)
    agent.process_step_data([obs, act, 0, next_obs, done])

    if done:
      agent.end_episode()
      obs = env.reset()
    else:
      obs = next_obs

    agent.end_step()
    step_idx += 1

  agent.end_task()

def train_task_model(agent, env, n_steps, print_rewards=False, print_freq=5000):
  agent.start_task(n_steps)

  obs = env.reset()
  step_idx = 0
  recent_rewards = []
  episode_rewards = []
  while step_idx < n_steps:
    act = agent.sample_act(obs)
    next_obs, reward, done, _ = env.step(act)
    episode_rewards.append(reward)
    agent.process_step_data([obs, act, reward, next_obs, done])

    if done:
      agent.end_episode()
      obs = env.reset()
      recent_rewards.append(np.sum(episode_rewards))
      episode_rewards = []
    else:
      obs = next_obs

    if print_rewards and step_idx > 0 and step_idx % print_freq == 0:
      print('Step: {} | Episodes: {} | Ep rewards: {:.4f}'.format(
        step_idx, len(recent_rewards), np.mean(recent_rewards)))
      recent_rewards = []

    agent.end_step()
    step_idx += 1

  agent.end_task()
