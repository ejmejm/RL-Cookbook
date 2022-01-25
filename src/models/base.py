import torch
from torch import nn

def create_gridworld_convs(n_channels=1):
  return nn.Sequential(
        nn.Conv2d(n_channels, 8, 4, 2),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1),
        nn.ReLU()
    )

def create_gridworld_upconvs(n_channels=1):
    return nn.Sequential(
        nn.ConvTranspose2d(16, 8, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, n_channels, 4, 2),
        nn.Conv2d(n_channels, n_channels, 3, 1, 1))

def create_atari_convs(n_channels=4):
  return nn.Sequential(
        nn.Conv2d(n_channels, 32, 5, 5, 0),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, 5, 0),
        nn.ReLU()
    )

def create_atari_upconvs(n_channels=4):
    return nn.Sequential(
        nn.ConvTranspose2d(64, 32, 5, 5, 0),
        nn.ReLU(),
        nn.ConvTranspose2d(32, n_channels, 5, 5, 0),
        nn.ReLU(),
        nn.Conv2d(n_channels, n_channels, 3, 1, 1))

def create_convs_from_obs_dim(obs_dim):
  if obs_dim[1] <= 32:
      return create_gridworld_convs(obs_dim[0])
  return create_atari_convs(obs_dim[0])

def create_upconvs_from_obs_dim(obs_dim):
  if obs_dim[1] <= 32:
      return create_gridworld_upconvs(obs_dim[0])
  return create_atari_upconvs(obs_dim[0])

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_acts):
        super().__init__()
        convs = create_convs_from_obs_dim(obs_dim)

        test_input = torch.zeros(1, *obs_dim)
        with torch.no_grad():
            self.encoder_output_size = convs(test_input).view(-1).shape[0]

        self.layers = nn.Sequential(
            convs,
            nn.Flatten(),
            nn.Linear(self.encoder_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_acts))

    def forward(self, x):
        return self.layers(x)

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        convs = create_convs_from_obs_dim(obs_dim)

        test_input = torch.zeros(1, *obs_dim)
        with torch.no_grad():
            self.encoder_output_size = convs(test_input).view(-1).shape[0]

        self.layers = nn.Sequential(
            convs,
            nn.Flatten(),
            nn.Linear(self.encoder_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.layers(x)

class SFNetwork(nn.Module):
    def __init__(self, obs_dim, embed_dim=256):
        super().__init__()
        convs = create_convs_from_obs_dim(obs_dim)

        test_input = torch.zeros(1, *obs_dim)
        with torch.no_grad():
            self.encoder_output_size = convs(test_input).view(-1).shape[0]
        
        self.encoder = nn.Sequential(
            convs,
            nn.Flatten(),
            nn.Linear(self.encoder_output_size, embed_dim),
            nn.LayerNorm(embed_dim))

        self.sf_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        embeds = self.encoder(x)
        sfs = self.sf_predictor(embeds)
        return embeds, sfs

class StatePredictionModel(nn.Module):
  def __init__(self, obs_dim, n_acts):
    super().__init__()
    self.downsample_convs = create_convs_from_obs_dim(obs_dim)

    test_input = torch.zeros([1] + list(obs_dim))
    output_dim = self.downsample_convs(test_input).view(-1).shape[0]

    self.fc = nn.Sequential(
        nn.Linear(output_dim + n_acts, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim),
        nn.ReLU())

    self.upsample_convs = create_upconvs_from_obs_dim(obs_dim)

    self.encoder = nn.Sequential(
        self.downsample_convs,
        nn.Flatten())

  def forward(self, obs, acts):
    conv_out = self.downsample_convs(obs)
    z = conv_out.view(obs.shape[0], -1)
    z = torch.cat([z, acts], dim=1)
    z = self.fc(z)
    z = z.view(*list(conv_out.shape))
    out = self.upsample_convs(z)
    return out