# Source: https://docs.wandb.ai/guides/track/public-api-guide

import pandas as pd 
import wandb

api = wandb.Api()
entity, project = 'ejmejm', 'rl_representation_learning-sweep_configs'  # set to your entity and project 
runs = api.runs(entity + '/' + project) 

HISTORY_VARS = ['env_task_step', 'env_task_episode_reward']

history_list, config_list, name_list, sweep_list = [], [], [], []
for run in runs:
    if run.state != 'finished':
        continue
    # .history contains the output keys/values for metrics like reward.
    #  We call ._json_dict to omit large files

    history_list.append(run.history(keys=HISTORY_VARS).to_dict())
    # print(run.state)
    # print(dir(run))
    print(run.name, run.id, [key for key in run.history().to_dict().keys() if 'gradient' not in key])

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

    # .sweep contains the sweep id.
    sweep_list.append(run.sweep.id)

runs_df = pd.DataFrame({
    'config': config_list,
    'name': name_list,
    'sweep': sweep_list,
    'history': history_list})

runs_df.to_csv('data/run_data.csv')

print('{}/{} runs saved.'.format(len(runs_df), len(runs)))