from collections import namedtuple
from typing import Iterable

import torch


TransitionData = namedtuple('TransitionData',
    ('obs', 'action', 'reward', 'next_obs', 'done'))

class RewardNormalizer():
    def __init__(self, hist_capacity=int(1e5)):
        self.hist_capacity = hist_capacity
        self.hist = []

    def _update(self, rewards):
        self.hist.extend(rewards)
        self.hist = self.hist[-self.hist_capacity:]

    def _normalize(self, rewards):
        hist = torch.tensor(self.hist, dtype=torch.float32,
            device=rewards.device)
        return (rewards - torch.mean(hist)) / torch.std(hist)

    def normalize(self, rewards):
        self._update(rewards)
        return self._normalize(rewards)