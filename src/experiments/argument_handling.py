import argparse

def make_arg_parser():
    # Create arguments for the environent, exporation agent, representation learner, and training agent
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Gridworld') # 'CrazyClimberNoFrameskip-v4'
    parser.add_argument('--exp_agent', type=str, default='EzExplore')
    parser.add_argument('--repr_learner', type=str, default='SFPredictor')
    parser.add_argument('--task_agent', type=str, default='Rainbow')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_steps', type=int, default=int(1e5))
    parser.add_argument('--task_steps', type=int, default=int(1e5))
    parser.add_argument('--freeze_encoder', default=False, action='store_true')
    parser.add_argument('--n_runs', type=int, default=1) # Does not work for sweeps

    parser.add_argument('--exp_agent_args', type=str, metavar='KEY=VALUE', nargs='+', default={})
    parser.add_argument('--exp_model_args', type=str, metavar='KEY=VALUE', nargs='+', default={})

    parser.add_argument('--repr_agent_args', type=str, metavar='KEY=VALUE', nargs='+', default={})
    parser.add_argument('--repr_model_args', type=str, metavar='KEY=VALUE', nargs='+', default={})

    parser.add_argument('--task_agent_args', type=str, metavar='KEY=VALUE', nargs='+', default={})
    parser.add_argument('--task_model_args', type=str, metavar='KEY=VALUE', nargs='+', default={})

    return parser

# Source: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
def parse_var(s):
    print(s)
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
            setattr(args, arg_name, parse_vars(getattr(args, arg_name)))
    return args

def make_and_parse_args():
    parser = make_arg_parser()
    args = parser.parse_args()
    args = format_args(args)
    return args