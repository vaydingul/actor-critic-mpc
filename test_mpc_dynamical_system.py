import env
from mpc import ModelPredictiveControl, ModelPredictiveControlSimple
from system import DynamicalSystem
from gymnasium.wrappers import FlattenObservation
from wrapper import RelativePosition
import gymnasium as gym
import numpy as np
import torch
from torch import nn

size = 20
agent_location_noise_level = 0.1
agent_velocity_noise_level = 0.01
target_location_noise_level = 0.1
target_velocity_noise_level = 0.01

# Create system
system = DynamicalSystem(
    size=size,
    random_force_probability=0.01,
    random_force_magnitude=10.0,
    friction_coefficient=0.25,
    wind_gust=[0.5, 0.5],
    wind_gust_region=[[0.4, 0.6], [0.4, 0.6]],
)

# Create environment
env = gym.make(
    "DynamicalSystem-v0",
    render_mode="human",
    size=size,
    distance_threshold=1.0,
    system=system,
    agent_location_noise_level=agent_location_noise_level,
    agent_velocity_noise_level=agent_velocity_noise_level,
    target_location_noise_level=target_location_noise_level,
    target_velocity_noise_level=target_velocity_noise_level,
)
env = FlattenObservation(env)

# Create Model Predictive Control model
mpc = ModelPredictiveControlSimple(
    system,
    size=size,
    lr=2.0,
    agent_location_noise_level=agent_location_noise_level,
    agent_velocity_noise_level=agent_velocity_noise_level,
    target_location_noise_level=target_location_noise_level,
    target_velocity_noise_level=target_velocity_noise_level,
    num_optimization_step=40,
)


while True:
    observation, _ = env.reset()
    observation = torch.Tensor(observation.copy()).unsqueeze(0)
    agent_location = observation[:, :2]
    agent_velocity = observation[:, 2:4]
    target_location = observation[:, 4:6]
    target_velocity = observation[:, 6:8]

    terminated = False

    while not terminated:
        action, _ = mpc(
            agent_location, agent_velocity, target_location, target_velocity
        )

        action_selected = action[0][0].detach().clone().numpy()

        observation, reward, terminated, truncated, information = env.step(
            action_selected
        )

        observation = torch.Tensor(observation.copy()).unsqueeze(0)
        agent_location = observation[:, :2]
        agent_velocity = observation[:, 2:4]
        target_location = observation[:, 4:6]
        target_velocity = observation[:, 6:8]

        print(f"Agent location: {agent_location}")
        print(f"Agent velocity: {agent_velocity}")
        print(f"Target location: {target_location}")
        print(f"Target velocity: {target_velocity}")
        print(f"Action: {action_selected}")

        env.render()
