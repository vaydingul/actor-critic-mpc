from typing import Any
import env
from mpc import ModelPredictiveControlWithoutOptimizer
from system import Acrobot, angle_normalize

# Import make_vec_env to allow parallelization
from stable_baselines3.common.env_util import make_vec_env
import torch
import numpy as np




def cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["theta_1"].shape
    device = predicted_state["theta_1"].device

    predicted_theta_1 = predicted_state["theta_1"]
    predicted_theta_2 = predicted_state["theta_2"]
    predicted_theta_1_dot = predicted_state["theta_1_dot"]
    predicted_theta_2_dot = predicted_state["theta_2_dot"]

    target_theta_1 = (
        target_state["theta_1"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )
    target_theta_1_dot = (
        target_state["theta_1_dot"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )
    target_theta_2 = (
        target_state["theta_2"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )
    target_theta_2_dot = (
        target_state["theta_2_dot"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )

    if cost_dict is None:
        cost_dict = dict(
            theta_1_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
            theta_1_dot_weight=torch.ones(
                batch_size, prediction_horizon, 1, device=device
            )
            * 0.0,
            theta_2_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
            theta_2_dot_weight=torch.ones(
                batch_size, prediction_horizon, 1, device=device
            )
            * 0.0,
            action_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
        )

    cost = (
        (
            torch.nn.functional.mse_loss(
                angle_normalize(predicted_theta_1),
                angle_normalize(target_theta_1),
                reduction="none",
            )
            * cost_dict["theta_1_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                angle_normalize(predicted_theta_2),
                angle_normalize(target_theta_2),
                reduction="none",
            )
            * cost_dict["theta_2_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                predicted_theta_1_dot,
                target_theta_1_dot,
                reduction="none",
            )
            * cost_dict["theta_1_dot_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                predicted_theta_2_dot,
                target_theta_2_dot,
                reduction="none",
            )
            * cost_dict["theta_2_dot_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.norm(
                action,
                p=2,
                dim=-1,
                keepdim=True,
            )
            * cost_dict["action_weight"]
        )
        .mean(1)
        .sum()
    )

    # cost = (
    #     torch.norm(
    #         - torch.cos(predicted_theta_1)
    #         - torch.cos(predicted_theta_2 + predicted_theta_1)
    #         - 1,
    #         p=2,
    #         dim=-1,
    #         keepdim=True,
    #     )
    #     .mean(1)
    #     .sum()
    # )

    return cost


def obs_to_state_target(obs) -> tuple[Any, Any]:
    theta_1 = torch.atan2(obs[:, 1], obs[:, 0]).unsqueeze(-1)
    theta_2 = torch.atan2(obs[:, 3], obs[:, 2]).unsqueeze(-1)
    theta_1_dot = obs[:, 4].unsqueeze(-1)
    theta_2_dot = obs[:, 5].unsqueeze(-1)

    state = dict(
        theta_1=theta_1,
        theta_2=theta_2,
        theta_1_dot=theta_1_dot,
        theta_2_dot=theta_2_dot,
    )

    target = dict(
        theta_1=torch.ones_like(theta_1) * np.pi,
        theta_2=torch.ones_like(theta_2) * 0.0,
        theta_1_dot=torch.ones_like(theta_1_dot) * 0.0,
        theta_2_dot=torch.ones_like(theta_2_dot) * 0.0,
    )

    return state, target


# Create system
system = Acrobot()

# Create environment
env = make_vec_env(
    "AcrobotContinuous-v0",
    n_envs=1,
    seed=42,
)


# Create Model Predictive Control model
mpc = ModelPredictiveControlWithoutOptimizer(
    system,
    cost,
    action_size=1,
    prediction_horizon=40,
    num_optimization_step=10,
    lr=1.0,
    device="cpu",
)


observation = env.reset()

observation = torch.Tensor(observation.copy())

state, target = obs_to_state_target(observation)


while True:
    action, cost_value = mpc(state, target)

    # print(action)

    action_ = action.clone().detach().numpy()
    action_selected = action_[:, 0]
    # print(action_selected.shape)
    # print(f"Action: {action_selected}")
    # print(f"Cost: {cost_value}")
    observation, reward, _, information = env.step(action_selected)
    # print(reward)
    observation = torch.Tensor(observation.copy())
    # print(f"State: {observation}")
    state, target = obs_to_state_target(observation)
    print(f"State: {state}")
    print(f"Target: {target}")
    env.render("human")
