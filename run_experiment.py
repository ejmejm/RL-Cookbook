import argparse
import copy
import sys

import torch
import wandb

sys.path.append('..')

from src.agents.register import *
from src.training import *
from src.envs.creation import make_env
from src.utils import make_arg_parser, format_args
from src.utils.constants import *

if __name__ == '__main__':
    # Parse arguments
    parser = make_arg_parser()
    args = parser.parse_args()
    args = format_args(args)
    args_dict = args.__dict__

    for _ in range(args.n_runs):
        # Init wandb
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=args_dict)

        env = make_env(args.env)
        device = torch.device(args.device)

        # Create representation learner
        repr_learner = create_repr_learner(args.repr_learner, env, args_dict)
        if repr_learner is not None and args.exp_agent != 'None':
            # Create exploration agent
            explore_agent = create_exploration_agent(args.exp_agent, env, args_dict, repr_learner)
            
            # Run exploration
            print('Starting exploration...')
            train_exploration_model(explore_agent, env, args.exp_steps)
            print('Exploration complete!')

            encoder = copy.deepcopy(repr_learner.encoder).to('cpu')
        else:
            encoder = None

        # Create task agent
        task_agent = create_task_agent(args.task_agent, env, args_dict, encoder, None)

        # Run training
        print('Starting task training...')
        train_task_model(task_agent, env, args.task_steps, print_rewards=True)
        print('Task training complete!')
        
        wandb.finish()