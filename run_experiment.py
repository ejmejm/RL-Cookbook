import argparse
import copy
import sys

import torch

sys.path.append('..')

from src.agents.register import *
from src.training import *
from src.envs.creation import make_env

# Create arguments for the environent, exporation agent, representation learner, and training agent
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Gridworld') # 'CrazyClimberNoFrameskip-v4'
parser.add_argument('--exp_agent', type=str, default='EzExplorerAgent')
parser.add_argument('--repr_learner', type=str, default='SFPredictor')
parser.add_argument('--task_agent', type=str, default='RainbowAgent')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--explore_steps', type=int, default=int(1e5))
parser.add_argument('--task_steps', type=int, default=int(1e5))

parser.add_argument('--exp_agent_args', type=str, metavar='KEY=VALUE', nargs='+', default={})
parser.add_argument('--exp_model_args', type=str, metavar='KEY=VALUE', nargs='+', default={})

parser.add_argument('--repr_agent_args', type=str, metavar='KEY=VALUE', nargs='+', default={})
parser.add_argument('--repr_model_args', type=str, metavar='KEY=VALUE', nargs='+', default={})

parser.add_argument('--task_agent_args', type=str, metavar='KEY=VALUE', nargs='+', default={})
parser.add_argument('--task_model_args', type=str, metavar='KEY=VALUE', nargs='+', default={})


# Source: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
def parse_var(s):
    items = s.split('=')
    key = items[0].strip()
    value = '='.join(items[1:])
    return (eval(key), eval(value))

def parse_vars(items):
    d = {}
    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

DICT_ARGS_LIST = ['exp_agent_args', 'exp_model_args', 'repr_agent_args',
    'repr_model_args', 'task_agent_args', 'task_model_args']
def format_args(args):
    for arg_name in DICT_ARGS_LIST:
        if len(getattr(args, arg_name)) > 0:
            args[arg_name] = parse_vars(args[arg_name])
    return args


if __name__ == '__main__':
    args = parser.parse_args()
    args = format_args(args)
    args_dict = args.__dict__

    env = make_env(args.env)
    device = torch.device(args.device)

    # Create representation learner
    repr_learner = create_repr_learner(args.repr_learner, env, args_dict)
    if repr_learner is not None and args.exp_agent != 'None':
        # Create exploration agent
        explore_agent = create_exploration_agent(args.exp_agent, env, args_dict, repr_learner)
        
        # Run exploration
        print('Starting exploration...')
        train_task_model(explore_agent, env, args.explore_steps, print_rewards=True)
        print('Exploration complete!')

        encoder = copy.deepcopy(repr_learner.encoder).to('cpu')
    else:
        encoder = None

    # Create task agent
    task_agent = create_task_agent(args.task_agent, env, args_dict, encoder, None)

    # Run training
    print('Starting task training...')
    train_task_model(task_agent, env, args.task_steps, print_rewards=False)
    print('Task training complete!')