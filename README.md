# Representations & Exploration For RL
*by Edan Meyer*

## Goals

This project aims to run a number of a number of experiments to determine the effectiveness of different future prediction objectives have on a learned representation. Below are the different dimensions we plan on testing

**Objectives**

Each objective will have two variants: one where the prediction error is minimized, and the other where the mutual information between the representation and item are maximized.

- Next state
- Successor features
- (Optional) Future state
- (Optional) Inverse dynamics (predicting action)
- (Optional) SR-NN agumentation (sparse representations)

**Exploration Method**

Given that the learning of representations is heavily dependent on their exploration, we believe that it is also necessary to test a number of exploration methods to see how they affect the learning representations.

- Surprisal based exploration (error in predicted representation)
- Entropy based exploration

**Environments**

- Gridworld
- Atari
- Gym hard exploration envs
- (Optional) DM Control Suite

**Testing Methodology**

There is no standardized method for testing the usefulness of learned representations, and different representations may perform better under different circumstances.

- Models
    - Efficient Rainbow
    - (Optional) PPO
    - (Optional) Dreamer V2
- Encoder weights
    - Frozen
    - Trainable
- Transfer method
    - Reuse learned encoder
    - Randomized with no transfer (baseline)

**Metrics**

- Extrinsic reward during exploration
- Extrinsic reward after downstream task training