from collections import namedtuple


TransitionData = namedtuple('TransitionData',
    ('obs', 'action', 'reward', 'next_obs', 'done'))