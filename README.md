# Actor-Critic Model Predictive Control (AC-MPC)

This repository implements an Actor-Critic Model Predictive Control (AC-MPC) framework that combines the strengths of Model Predictive Control (MPC) with Reinforcement Learning (RL). The implementation includes various classic control environments and a novel dynamical system environment.

## Project Overview

The project implements a hybrid control approach that integrates:
- Model Predictive Control (MPC) for optimal trajectory planning
- Reinforcement Learning (specifically PPO - Proximal Policy Optimization) for policy learning
- Actor-Critic architecture for improved policy optimization

## Environments

The following environments are supported:
1. Pendulum
2. Cart Pole (both discrete and continuous reward versions)
3. Mountain Car
4. Acrobot
5. Custom Dynamical System (with configurable parameters like wind gusts, friction, etc.)

## Key Components

### Core Modules
- `mpc.py`: Implementation of Model Predictive Control algorithms
- `policy.py`: Policy networks and Actor-Critic implementations
- `system.py`: System dynamics and models
- `env.py`: Environment implementations and wrappers
- `costs.py`: Cost functions for different environments
- `linearizer.py`: System linearization utilities
- `utils.py`: Utility functions and helpers

### Training Scripts
- Training scripts for each environment (e.g., `train_acmpc_multienv_pendulum_args.py`)
- PPO baseline training scripts (e.g., `train_ppo_pendulum.py`)
- Support for multi-environment training

### Testing Scripts
- Test scripts for each environment (e.g., `test_mpc_pendulum.py`)
- Systematic evaluation scripts (e.g., `test_acmpc_systematic_dynamical_system_args.py`)

## Features

- Hybrid control combining MPC and RL
- Support for multiple classic control environments
- Configurable system parameters (friction, wind gusts, etc.)
- Gaussian noise wrappers for robustness
- Integration with Weights & Biases for experiment tracking
- TensorBoard support for visualization
- Systematic evaluation tools

## Installation

1. Create a conda environment using the provided environment files:
   ```bash
   # For MacOS/Linux
   conda env create -f environment.yml
   # For Linux-specific dependencies
   conda env create -f environment_linux.yml
   ```

2. Activate the environment:
   ```bash
   conda activate acmpc
   ```

## Usage

### Training

Train an AC-MPC agent on different environments:
```bash
# Train on Pendulum
python train_acmpc_multienv_pendulum_args.py

# Train on Cart Pole
python train_acmpc_multienv_cart_pole_args.py

# Train on Mountain Car
python train_acmpc_multienv_mountain_car_args.py
```

### Testing

Test trained models:
```bash
# Test on Pendulum
python test_acmpc_pendulum_args.py --model_name path/to/model

# Test on Dynamical System
python test_acmpc_dynamical_system_args.py --model_name path/to/model
```

## Key Parameters

- `n_envs`: Number of parallel environments
- `prediction_horizon`: MPC prediction horizon
- `num_optimization_step`: Number of optimization steps in MPC
- `gaussian_noise_scale`: Scale of Gaussian noise for robustness
- Environment-specific parameters (e.g., wind gusts, friction coefficients)

## Experiment Tracking

The project uses Weights & Biases for experiment tracking. Key metrics logged include:
- Training rewards
- Episode lengths
- Policy gradients
- Value function losses
- Model checkpoints

## Dependencies

Key dependencies include:
- PyTorch
- Stable Baselines3
- Gymnasium
- TensorBoard
- Weights & Biases

For a complete list of dependencies, refer to `environment.yml` or `environment_linux.yml`. 
