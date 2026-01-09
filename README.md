# Learning Reinforcement Learning

A collection of Jupyter notebooks implementing core Reinforcement Learning (RL) concepts and algorithms using Gymnasium.

## Contents

- **`01_Basic_RL.ipynb`**: Introduction to MDPs and Dynamic Programming.
  - Policy Evaluation & Improvement
  - Policy Iteration (PI)
  - Value Iteration (VI)
  - Environment: `FrozenLake-v1`

- **`02_Multi_Arm_Bandits.ipynb`**: Exploration vs. Exploitation strategies.
  - Pure Exploitation/Exploration
  - Optimistic Initial Values
  - $\epsilon$-greedy strategy
  - Upper Confidence Bound (UCB)
  - Thompson Sampling
  - Environment: `BanditTenArmedGaussian-v0`

## Environment Setup

The project uses conda environment. Key dependencies include:
- `numpy`
- `gymnasium`
- `matplotlib`
- `tqdm`
- `gym-bandits`
- `gym-walk`
