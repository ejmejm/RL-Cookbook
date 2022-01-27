from collections import namedtuple
from typing import Iterable

import numpy as np
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
        return (rewards - torch.mean(hist)) / (torch.std(hist) + 1e-7)

    def normalize(self, rewards):
        self._update(rewards)
        return self._normalize(rewards)

# Tracks the entropy of a procedural discrete distribution
class DiscreteEntropyTracker():
    def __init__(self, dim, decay=0.97):
        self.running_counts = [0 for _ in range(dim)]
        self.decay = decay

    def calc_entropy(self, sample_idx=None):
        if sample_idx is not None:
            for i in range(len(self.running_counts)):
                if i == sample_idx:
                    self.running_counts[i] = \
                        self.decay * self.running_counts[i] + \
                        (1 - self.decay)
                else:
                    self.running_counts[i] *= self.decay
        probs = np.array(self.running_counts) / sum(self.running_counts)
        return -sum([p * np.log(p + 1e-7) for p in probs])