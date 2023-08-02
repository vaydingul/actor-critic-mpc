import time
from typing import Any

import numpy as np
import env
from mpc import ModelPredictiveControlWithoutOptimizer
from system import CartPole
import gymnasium as gym
# Import make_vec_env to allow parallelization
from stable_baselines3.common.env_util import make_vec_env
import torch


def cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["x"].shape
    device = predicted_state["x"].device

    predicted_x = predicted_state["x"]
    predicted_x_dot = predicted_state["x_dot"]
    predicted_theta = predicted_state["theta"]
    predicted_theta_dot = predicted_state["theta_dot"]

    target_x = target_state["x"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    target_x_dot = target_state["x_dot"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    target_theta = target_state["theta"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    target_theta_dot = (
        target_state["theta_dot"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )

    if cost_dict is None:
        cost_dict = dict(
            x_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 10.0,
            x_dot_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
            theta_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 1000.0,
            theta_dot_weight=torch.ones(
                batch_size, prediction_horizon, 1, device=device
            )
            * 0.0,
            action_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
        )

    cost = (
        (
            torch.nn.functional.mse_loss(
                predicted_x,
                target_x,
                reduction="none",
            )
            * cost_dict["x_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                predicted_x_dot,
                target_x_dot,
                reduction="none",
            )
            * cost_dict["x_dot_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                predicted_theta,
                target_theta,
                reduction="none",
            )
            * cost_dict["theta_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                predicted_theta_dot,
                target_theta_dot,
                reduction="none",
            )
            * cost_dict["theta_dot_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (((action).pow(2)) * cost_dict["action_weight"]).mean(1).sum()

    return cost


def obs_to_state_target(obs) -> tuple[Any, Any]:
    x = obs[:, 0].unsqueeze(1)
    x_dot = obs[:, 1].unsqueeze(1)
    theta = obs[:, 2].unsqueeze(1)
    theta_dot = obs[:, 3].unsqueeze(1)

    state = dict(
        x=x,
        x_dot=x_dot,
        theta=theta,
        theta_dot=theta_dot,
    )

    target = dict(
        x=torch.ones_like(x) * 0.0,
        x_dot=torch.ones_like(x_dot) * 0.0,
        theta=torch.ones_like(theta) * 0.0,
        theta_dot=torch.ones_like(theta_dot) * 0.0,
    )

    return state, target


# Create system
system = CartPole()

# Create environment
env = make_vec_env(
    "CartPoleContinuous-v0",
    n_envs=1,
    seed=42,
)

# Create Model Predictive Control model
mpc = ModelPredictiveControlWithoutOptimizer(
    system,
    cost,
    action_size=1,
    prediction_horizon=10,
    num_optimization_step=10,
    lr=1.0,
    std=2.5,
    device="cpu",
)


observation = env.reset()

observation = torch.Tensor(observation.copy())

state, target = obs_to_state_target(observation)

counter = 0
while True:
    action, cost_value = mpc(state, target)

    action_ = action.clone().detach().numpy()
    action_selected = action_[:, 0]

    print(f"Action: {action_selected}")
    print(f"Cost: {cost_value}")
    observation, reward, done, information = env.step(action_selected)
    counter += 1
    print(f"Reward: {reward}")
    if done:
        print(f"Counter: {counter}")
        counter = 0
        time.sleep(5.0)
        

    observation = torch.Tensor(observation.copy())

    state, target = obs_to_state_target(observation)
    
    # time.sleep(0.1)
    env.render("human")
