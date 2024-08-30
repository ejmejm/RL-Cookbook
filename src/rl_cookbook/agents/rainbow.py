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
  """Rainbow DQN agent implementation.

  This agent implements the Rainbow DQN algorithm, which combines several
  improvements to the original DQN algorithm.

  Args:
    env: The environment to interact with.
    args: Configuration arguments for the agent.
    custom_encoder: Optional custom encoder network.
    repr_learner: Optional representation learner.
    tracked: Whether to track the model with wandb.
  """

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
    """Initializes the agent for a new task.

    Args:
      n_steps: Number of steps for the task.
    """
    # Reset metrics and seeds
    self.metrics = {'steps': [], 'rewards': [], 'Qs': [],
                    'best_avg_reward': -float('inf')}
    self.all_rewards = []
    self.ep_rewards = []

    self.mem = RainbowMemory(self.args, self.args.memory_capacity, self.obs_dim)
    self.priority_weight_increase = (1 - self.args.priority_weight) \
      / (n_steps - self.args.learn_start)

    # Construct validation memory
    self.val_mem = RainbowMemory(self.args, self.args.evaluation_size, self.obs_dim)

    self.last_update_episode = 0
    self.step_idx = 1

  def train_representation(self):
    raise NotImplementedError('Representation learning not implemented with Rainbow!')

  def sample_act(self, obs):
    """Samples an action from the agent's policy.

    Args:
      obs: The current observation.

    Returns:
      The selected action.
    """
    if self.step_idx % self.args.replay_frequency == 0:
      self.dqn.reset_noise() # Draw a new set of noisy weights

    if obs.device != self.args.device:
      obs = obs.to(self.args.device)
    action = self.dqn.act(obs) # Choose an action greedily (with noisy weights)
    return action

  def append_buffer(self, transition_data):
    """Appends a transition to the replay buffer.

    Args:
      transition_data: A tuple containing (obs, action, reward, _, done).
    """
    obs, action, reward, _, done = transition_data
    # Clip rewards
    if self.args.reward_clip > 0:
      reward = max(min(reward, self.args.reward_clip), -self.args.reward_clip)
    self.mem.append(obs, action, reward, done) # Append transition to memory

  def process_step_data(self, transition_data):
    """Processes data from a single environment step."""
    self.ep_rewards.append(transition_data[2])
    self.append_buffer(transition_data)

  def end_episode(self):
    """Performs end-of-episode operations."""
    self.all_rewards.append(sum(self.ep_rewards))
    self.ep_rewards = []

  def train(self):
    """Performs a single training step.

    Returns:
      The loss value from the training step.
    """
    loss = self.dqn.learn(self.mem)
    wandb.log({'task_agent_loss': loss})
    return loss

  def end_step(self):
    """Performs end-of-step operations, including training and updates."""
    if self.step_idx >= self.args.learn_start:
      # Anneal importance sampling weight β to 1
      self.mem.priority_weight = min(self.mem.priority_weight + \
                                      self.priority_weight_increase, 1)

      # Train with n-step distributional double-Q learning
      if self.step_idx % self.args.replay_frequency == 0:
        self.train()

      # Update target network
      if self.step_idx % self.args.target_update == 0:
        self.dqn.update_target_net()

      # Train representation network
      if self.repr_learner is not None and \
          self.step_idx % self.repr_learner.update_freq == 0:
        self.train_representation()

    self.step_idx += 1

  def train_mode(self):
    """Sets the agent to training mode."""
    self.dqn.train()

  def eval_mode(self):
    """Sets the agent to evaluation mode."""
    self.dqn.eval()