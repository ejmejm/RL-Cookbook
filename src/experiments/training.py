import torch

from .register import *
from .argument_handling import *
from ..envs.creation import make_env
from ..envs.simulation import *
from ..utils.constants import *


def train_loop(args):
    args.device = torch.device(args.device)
    args_dict = args.__dict__
    env = make_env(args.env)

    # Create representation learner
    repr_learner = create_repr_learner(args.repr_learner, env, args_dict)
    if repr_learner is not None and args.exp_agent.lower() != 'none':
        # Create exploration agent
        explore_agent = create_exploration_agent(args.exp_agent, env, args_dict, repr_learner)
        
        # Run exploration
        print('Starting exploration...')
        train_exploration_model(explore_agent, env, args.exp_steps, print_freq=args.reward_print_freq)
        print('Exploration complete!')

        encoder = copy.deepcopy(repr_learner.encoder).to('cpu')

        if args.freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
    else:
        encoder = None

    # Create task agent
    task_agent = create_task_agent(args.task_agent, env, args_dict, encoder, None)

    # Run training
    print('Starting task training...')
    train_task_model(task_agent, env, args.task_steps, print_freq=args.reward_print_freq)
    print('Task training complete!')