from torch import nn

from .agents import RainbowAgent, NextStatePredReprLearner, TestRL
from .agents.Rainbow import DEFAULT_RAINBOW_ARGS
from .envs import *
from .training import *

def create_small_convs(input_dim):
  return nn.Sequential(
        nn.Conv2d(input_dim, 8, 4, 2),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1),
        nn.ReLU(),
        nn.Flatten()
    )

class FuturePredictor(nn.Module):
  def __init__(self, input_dim, n_acts):
    super().__init__()
    self.downsample_convs = nn.Sequential(
        nn.Conv2d(input_dim[0], 8, 4, 2),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1),
        nn.ReLU())

    test_input = torch.zeros([1] + list(input_dim))
    output_dim = self.downsample_convs(test_input).view(-1).shape[0]

    self.fc = nn.Sequential(
        nn.Linear(output_dim + n_acts, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim),
        nn.ReLU())

    self.upsample_convs = nn.Sequential(
        nn.ConvTranspose2d(16, 8, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, input_dim[0], 4, 2))

  def forward(self, obs, acts):
    conv_out = self.downsample_convs(obs)
    z = conv_out.view(obs.shape[0], -1)
    z = torch.cat([z, acts], dim=1)
    z = self.fc(z)
    z = z.view(*list(conv_out.shape))
    out = self.upsample_convs(z)
    return out


env = create_simple_gridworld_env(True, 100)
# env = create_crazy_climber_env()

fp = FuturePredictor(list(env.observation_space.shape), env.action_space.n)
# out = fp(torch.zeros([2, 1, 16, 16]), torch.zeros([2, 4]))
# print(out.shape)

repr_learner = NextStatePredReprLearner(fp)


custom_encoder = None
if env.observation_space.shape[1] <= 42:
  custom_encoder = create_small_convs(env.observation_space.shape[0])

agent = RainbowAgent(DEFAULT_RAINBOW_ARGS, env, custom_encoder, repr_learner)
# agent = TestRL(agent)

# agent.start_task(1000)
# obs = env.reset()
# act = agent.sample_act(obs)
# print('Act:', act)
# obs, reward, done, _ = env.step(act)
# agent.process_step_data((obs, act, reward, obs, done))
# agent.end_step()

train_task_model(agent, env, int(1e5))