# The (My) RL Cookbook
*Author: Edan Meyer*


This repository implements and collates a few baseline RL algorithms, environments, and a simple testing harnesses that I use for evaluating the effectiveness of learned representations in RL.
It is not meant to be comprhensive, but rather to serve as a baseline for my work. Only the core ideas that are shared throughout my various projects in representation learning are implemented.
The implementations are intentionally minimal, but should also be easy to understand and extend.


# Getting Started

To use this repository for your own experiments:

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   pip install -e .
   ```

2. Update the `utils/constants.py` folder with your own Weights & Biases details.

3. Run experiments with the `run_experiment.py` script.
    - Example of a single run with DQN in GridWorld over 100K environment steps:
        ```bash
        python run_experiment.py --env Gridworld --agent DQN --n_steps 100000
        ```
    - You can also run hyperparameter sweeps by following [Wandb's guide to sweeps](https://docs.wandb.ai/guides/sweeps) and using the `sweep_agent.py` script as your entry point.
    - All additional arguments can be found in the `experiments/argument_handling.py` file.

4. Explore notebooks for example usecases:
   - `VISR.ipynb`: Implements and demonstrates the [VISR algorithm](https://arxiv.org/abs/1906.05030)
   - `goal_conditioned_policies.ipynb`: Experiments with goal-conditioned policies

5. Customize experiments:
   - Modify `experiments/argument_handling.py` to add new command-line arguments
   - Create new RL algorithms in the `agents/` directory
   - Add new architectures in the `models/` directory
   - Implement new environments in the `envs/` directory
   - Modify the training loop in the `experiments/` directory

6. Visualize results:
   - Results are automatically logged to Weights & Biases (wandb)
   - Access your experiment dashboard at wandb.ai
   - You can also use the `results/import_results.py` script to download data from wandb and visualize it locally


# Understanding the Code Base

This projects separates agents into 3 categories that can be mixed and matched: the **RL algorithm**, the **exploration method**, and optionally an **auxiliary task** for representation learning. The agent architecture is split into an encoder and policy/value heads. Auxiliary tasks can be optionally used to learn representations for the encoder. The RL algorithm backpropagates gradients through the policy and value heads, and optionally through the encoder.

**RL Algorithms**
- DQN
- Double and Dueling DQN variants
- Efficient Rainbow
- PPO

**Environments**
- Gridworld
- Atari
- Gym hard exploration envs
- Procgen

**Exploration Method**
- Epsilon greedy
- Ez explore (epsilon greedy + sticky actions)
- Surprisal based exploration (error in auxiliary task predictions)
- Entropy based exploration

**Auxiliary Tasks** (*for learning representations*)
- Successor feature based predictions (TD(0)-error of predicted successor features)
- Next observation predictions (MSE of predicted next observation)

## License

This project is licensed under the MIT License. Please see the attached license file for more details.