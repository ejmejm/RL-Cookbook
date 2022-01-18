import copy
from types import SimpleNamespace

import gym
from torch import nn

from .base import BaseRepresentationLearner
from .representation import NextStatePredictor, SFPredictor
from .exploration import EzExplorerAgent, MaxEntropyExplorerAgent, SurprisalExplorerAgent
from .rainbow import RainbowAgent
from .Rainbow import DEFAULT_RAINBOW_ARGS
from ..models import SFNetwork, PolicyNetwork, CriticNetwork, StatePredictionModel


REPR_LEARNERS = {
    'NextStatePredictor': lambda env, args:
        NextStatePredictor(
            StatePredictionModel(list(env.observation_space.shape), env.action_space.n) \
                .to(args['device']),
            **args['repr_agent_args']),
    'SFPredictor': lambda env, args:
        SFPredictor(
            SFNetwork(list(env.observation_space.shape),
                **args['repr_model_args']).to(args['device']),
            **args['repr_agent_args']),
    'None': lambda env, args: None
}

EXPLORATION_AGENTS = {
    'EzExplore': lambda env, args, repr_learner: 
        EzExplorerAgent(env, repr_learner=repr_learner, **args['exp_agent_args']),
    'Surprisal': lambda env, args, repr_learner:
        SurprisalExplorerAgent(
            env,
            PolicyNetwork(list(env.observation_space.shape), env.action_space.n) \
                .to(args['device']),
            CriticNetwork(list(env.observation_space.shape)).to(args['device']),
            repr_learner,
            **args['exp_agent_args']),
    'MaxEntropy': lambda env, args, repr_learner:
        MaxEntropyExplorerAgent(
            env,
            PolicyNetwork(list(env.observation_space.shape), env.action_space.n) \
                .to(args['device']),
            CriticNetwork(list(env.observation_space.shape)).to(args['device']),
            repr_learner,
            device = args['device'],
            **args['exp_agent_args'])
}

TASK_AGENTS = {
    'Rainbow': lambda env, args, encoder, repr_learner:
        RainbowAgent(
            env,
            dict_to_args(combine_args(DEFAULT_RAINBOW_ARGS.__dict__,
                args['task_model_args'], {'device': args['device']})),
            encoder, repr_learner)
}

def combine_args(*args):
    combined_args = copy.copy(args[0])
    for arg_set in args[1:]:
        combined_args.update(arg_set)
    return combined_args

def dict_to_args(d):
    args = SimpleNamespace()
    for k, v in d.items():
        setattr(args, k, v)
    return args
    
def create_repr_learner(repr_learner_name: str, env: gym.Env, args: dict):
    return REPR_LEARNERS[repr_learner_name](env, args)

def create_exploration_agent(exploration_agent_name: str, env: gym.Env, args: dict,
                             repr_learner: BaseRepresentationLearner):
    return EXPLORATION_AGENTS[exploration_agent_name](env, args, repr_learner)

def create_task_agent(task_agent_name: str, env: gym.Env, args: dict,
                      encoder: nn.Module, repr_learner: BaseRepresentationLearner):
    return TASK_AGENTS[task_agent_name](env, args, encoder, repr_learner)