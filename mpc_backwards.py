import env
from mpc import ModelPredictiveControl
from system import DynamicalSystem
from gymnasium.wrappers import FlattenObservation
from wrapper import RelativePosition
import gymnasium as gym
import numpy as np
import torch
from torch import nn

size = 10
agent_location_noise_level = 0.1
agent_velocity_noise_level = 0.01
target_location_noise_level = 0.1
target_velocity_noise_level = 0.01

# Create system
system = DynamicalSystem(
    size=size,
    random_force_probability=0.01,
    random_force_magnitude=10.0,
    friction_coefficient=0.1,
    wind_gust=[0.5, 0.5],
    wind_gust_region=[[0.3, 0.7], [0.3, 0.7]],
)

# Create environment
env = gym.make(
    "DynamicalSystem-v0",
    render_mode="rgb_array",
    size=size,
    distance_threshold=0.5,
    system=system,
    agent_location_noise_level=agent_location_noise_level,
    agent_velocity_noise_level=agent_velocity_noise_level,
    target_location_noise_level=target_location_noise_level,
    target_velocity_noise_level=target_velocity_noise_level,
    force_penalty_level=0.1,
)
env = FlattenObservation(env)
observation, _ = env.reset()

# Create Model Predictive Control model
mpc = ModelPredictiveControl(
    system,
    size=size,
    lr=0.1,
    agent_location_noise_level=agent_location_noise_level,
    agent_velocity_noise_level=agent_velocity_noise_level,
    target_location_noise_level=target_location_noise_level,
    target_velocity_noise_level=target_velocity_noise_level,
    num_optimization_step=30,
)
mpc.reset()


agent_location = torch.Tensor(observation[:2]).requires_grad_(True)
agent_velocity = torch.Tensor(observation[2:4])
target_location = torch.Tensor(observation[4:6])
target_velocity = torch.Tensor(observation[6:8])

action = mpc(agent_location, agent_velocity, target_location, target_velocity)

action_sum = action.sum()

# Differentiate action_sum with respect to agent_location
torch.autograd.grad(action_sum, agent_location)
