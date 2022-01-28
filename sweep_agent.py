import argparse
import sys

import wandb

sys.path.append('..')

from src.experiments.argument_handling import make_and_parse_args, format_args
from src.experiments.training import train_loop


if __name__ == '__main__':
    args = make_and_parse_args()
    wandb.init(config=args)
    args = format_args(argparse.Namespace(**wandb.config))
    train_loop(args)