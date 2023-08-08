import torch
import numpy as np
from typing import Any


def acrobot_cost(predicted_state, target_state, action=None, cost_dict=None):
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

    # cost = (
    #     (
    #         torch.nn.functional.mse_loss(
    #             wrap(predicted_theta_1),
    #             wrap(target_theta_1),
    #             reduction="none",
    #         )
    #         * cost_dict["theta_1_weight"]
    #     )
    #     .mean(1)
    #     .sum()
    # )

    # cost += (
    #     (
    #         torch.nn.functional.mse_loss(
    #             wrap(predicted_theta_2),
    #             wrap(target_theta_2),
    #             reduction="none",
    #         )
    #         * cost_dict["theta_2_weight"]
    #     )
    #     .mean(1)
    #     .sum()
    # )

    # cost += (
    #     (
    #         torch.nn.functional.mse_loss(
    #             predicted_theta_1_dot,
    #             target_theta_1_dot,
    #             reduction="none",
    #         )
    #         * cost_dict["theta_1_dot_weight"]
    #     )
    #     .mean(1)
    #     .sum()
    # )

    # cost += (
    #     (
    #         torch.nn.functional.mse_loss(
    #             predicted_theta_2_dot,
    #             target_theta_2_dot,
    #             reduction="none",
    #         )
    #         * cost_dict["theta_2_dot_weight"]
    #     )
    #     .mean(1)
    #     .sum()
    # )

    predicted_height = -(
        torch.cos(predicted_theta_1) + torch.cos(predicted_theta_1 + predicted_theta_2)
    )

    target_height = torch.tensor(1.0)

    cost = (
        (
            torch.nn.functional.mse_loss(
                predicted_height,
                target_height,
                reduction="none",
            )
            * 1.0
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

    return cost


def acrobot_obs_to_state_target(obs) -> tuple[Any, Any]:
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


def cart_pole_cost(predicted_state, target_state, action=None, cost_dict=None):
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


def cart_pole_obs_to_state_target(obs) -> tuple[Any, Any]:
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


def dynamical_system_cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["agent_location"].shape
    device = predicted_state["agent_location"].device

    predicted_agent_location = predicted_state["agent_location"]
    predicted_agent_velocity = predicted_state["agent_velocity"]

    target_agent_location = target_state["agent_location"].unsqueeze(1)
    target_agent_velocity = target_state["agent_velocity"].unsqueeze(1)

    if cost_dict is None:
        cost_dict = dict(
            location_weight=torch.ones(batch_size, prediction_horizon, device=device)
            * 1.0,
            velocity_weight=torch.ones(batch_size, prediction_horizon, device=device)
            * 0.1,
            action_first_derivative_weight=torch.ones(
                batch_size, prediction_horizon - 1, device=device
            )
            * 0.0,
            action_second_derivative_weight=torch.ones(
                batch_size, prediction_horizon - 2, device=device
            )
            * 0.0
            if prediction_horizon > 2
            else None,
        )

    # Location cost
    cost = (
        (
            cost_dict["location_weight"]
            * torch.norm(predicted_agent_location - target_agent_location, p=2, dim=-1)
        )
        .mean(dim=1)
        .sum()
    )

    # Velocity cost
    cost += (
        (
            cost_dict["velocity_weight"]
            * torch.norm(predicted_agent_velocity - target_agent_velocity, p=2, dim=-1)
        )
        .mean(dim=1)
        .sum()
    )

    return cost


def dynamical_system_obs_to_state_target(obs) -> tuple[Any, Any]:
    agent_location = obs[..., 4:6]
    agent_velocity = obs[..., 6:8]
    target_location = obs[..., 8:10]
    target_velocity = obs[..., 10:12]

    state = dict(
        agent_location=agent_location,
        agent_velocity=agent_velocity,
        target_location=target_location,
        target_velocity=target_velocity,
    )

    target = dict(
        agent_location=target_location,
        agent_velocity=target_velocity,
        target_location=None,
        target_velocity=None,
    )

    return state, target


def mountain_car_cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["position"].shape
    device = predicted_state["position"].device

    predicted_position = predicted_state["position"]
    predicted_velocity = predicted_state["velocity"]

    target_position = (
        target_state["position"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )
    target_velocity = (
        target_state["velocity"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )

    if cost_dict is None:
        cost_dict = dict(
            position_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.1,
            velocity_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 1.0,
            action_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.01,
        )

    cost = (
        (
            torch.nn.functional.mse_loss(
                predicted_position,
                target_position,
                reduction="none",
            )
            * cost_dict["position_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                torch.abs(predicted_velocity),
                target_velocity,
                reduction="none",
            )
            * cost_dict["velocity_weight"]
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

    return cost


def mountain_car_obs_to_state_target(obs) -> tuple[Any, Any]:
    position = obs[..., 0].unsqueeze(-1)
    velocity = obs[..., 1].unsqueeze(-1)

    state = dict(
        position=position,
        velocity=velocity,
    )

    target = dict(
        position=torch.ones_like(position) * 0.45,
        velocity=torch.ones_like(velocity) * 2.0,
    )

    return state, target


def pendulum_cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["theta"].shape
    device = predicted_state["theta"].device

    predicted_theta = predicted_state["theta"]
    predicted_theta_dot = predicted_state["theta_dot"]

    target_theta = target_state["theta"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    target_theta_dot = (
        target_state["theta_dot"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )

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
    elif isinstance(cost_dict, torch.Tensor):
        cost_dict = cost_tensor_to_dict(cost_dict)
    else:
        assert isinstance(cost_dict, dict) or isinstance(cost_dict, list)

    # cost = (
    #     (
    #         ((angle_normalize(predicted_theta) - angle_normalize(target_theta)).pow(2))
    #         * cost_dict["theta_weight"]
    #     )
    #     .mean(1)
    #     .sum()
    # )
    # cost += (
    #     (
    #         ((predicted_theta_dot - target_theta_dot).pow(2))
    #         * cost_dict["theta_dot_weight"]
    #     )
    #     .mean(1)
    #     .sum()
    # )
    # cost += (((action).pow(2)) * cost_dict["action_weight"]).mean(1).sum()

    cost = (
        (
            torch.nn.functional.mse_loss(
                angle_normalize(predicted_theta),
                angle_normalize(target_theta),
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

    return cost


def pendulum_obs_to_state_target(obs) -> tuple[Any, Any]:
    theta = torch.atan2(obs[:, 1], obs[:, 0]).unsqueeze(-1)
    theta_dot = obs[:, 2].unsqueeze(-1)

    state = dict(
        theta=theta,
        theta_dot=theta_dot,
    )

    target = dict(
        theta=torch.ones_like(theta) * 0.0,
        theta_dot=torch.ones_like(theta_dot) * 0.0,
    )

    return state, target
