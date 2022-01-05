from ..agents import BaseAgent

def train_exploration_model(agent: BaseAgent, env, n_steps):
  agent.start_task(n_steps)

  obs = env.reset()
  step_idx = 0
  while step_idx < n_steps:
    act = agent.sample_act(obs)
    next_obs, _, done, _ = env.step(act)
    agent.process_step_data([obs, act, None, next_obs, done])

    if done:
      agent.end_episode()
      obs = env.reset()
    else:
      obs = next_obs

    agent.end_step()
    step_idx += 1

  agent.end_task()

def train_task_model(agent, env, n_steps):
  agent.start_task(n_steps)

  obs = env.reset()
  step_idx = 0
  while step_idx < n_steps:
    act = agent.sample_act(obs)
    next_obs, reward, done, _ = env.step(act)
    agent.process_step_data([obs, act, reward, next_obs, done])

    if done:
      agent.end_episode()
      obs = env.reset()
    else:
      obs = next_obs

    agent.end_step()
    step_idx += 1

  agent.end_task()
