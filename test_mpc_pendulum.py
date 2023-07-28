import time
from typing import Any
import env
from mpc import ModelPredictiveControlWithoutOptimizer
from system import DynamicalSystem, Pendulum, angle_normalize
from gymnasium.wrappers import FlattenObservation
from wrapper import RelativePosition
import gymnasium as gym
import numpy as np
import torch
from torch import nn


def cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["theta"].shape
    device = predicted_state["theta"].device

    predicted_theta = predicted_state["theta"]
    predicted_theta_dot = predicted_state["theta_dot"]

    target_theta = target_state["theta"].unsqueeze(1)
    target_theta_dot = target_state["theta_dot"].unsqueeze(1)

    if cost_dict is None:
        cost_dict = dict(
            theta_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 10.0,
            theta_dot_weight=torch.ones(
                batch_size, prediction_horizon, 1, device=device
            )
            * 0.1,
            action_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.001,
        )

    cost = torch.tensor(0.0, device=device)

    cost += (
        (
            ((angle_normalize(predicted_theta) - angle_normalize(target_theta)).pow(2))
            * cost_dict["theta_weight"]
        )
        # .mean(1)
        .sum()
    )
    cost += (
        (
            (predicted_theta_dot - target_theta_dot).pow(2)
            * cost_dict["theta_dot_weight"]
        )
        # .mean(1)
        .sum()
    )

    cost += (action.pow(2) * cost_dict["action_weight"]).mean(1).sum()

    return cost


def obs_to_state_target(obs) -> tuple[Any, Any]:
    theta = torch.atan2(obs[:, 1], obs[:, 0]).unsqueeze(-1)
    theta_dot = obs[:, 2].unsqueeze(-1)

    state = dict(
        theta=theta,
        theta_dot=theta_dot,
    )

    target = dict(
        theta=torch.ones_like(theta) * 0,
        theta_dot=torch.ones_like(theta_dot) * 0.0,
    )

    return state, target


# Create system
system = Pendulum(
    dt=0.05,
    m=1.0,
    g=10.0,
    l=1.0,
)

# Create environment
env = gym.make(
    "Pendulum-v1",
    render_mode="human",
    g=10,
)


# Create Model Predictive Control model
mpc = ModelPredictiveControlWithoutOptimizer(
    system,
    cost,
    action_size=1,
    prediction_horizon=10,
    num_optimization_step=40,
    lr=1.0,
    device="cpu",
)


while True:
    observation, _ = env.reset()

    observation = torch.Tensor(observation.copy()).unsqueeze(0)

    state, target = obs_to_state_target(observation)

    terminated = False

    while not terminated:
        action, cost_value = mpc(state, target)

        action_selected = action[0][0].detach().clone().numpy()

        print(f"Action: {action_selected}")
        print(f"Cost: {cost_value}")
        observation, reward, terminated, truncated, information = env.step(
            action_selected
        )
        print(reward)
        observation = torch.Tensor(observation.copy()).unsqueeze(0)

        state, target = obs_to_state_target(observation)
        print(f"State: {state}")

        # time.sleep(0.1)
        env.render()
