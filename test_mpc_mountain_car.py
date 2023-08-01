from typing import Any
import env
from mpc import ModelPredictiveControlWithoutOptimizer
from system import MountainCar

# Import make_vec_env to allow parallelization
from stable_baselines3.common.env_util import make_vec_env
import torch


def cost(predicted_state, target_state, action=None, cost_dict=None):
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
            * 0.1,
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


def obs_to_state_target(obs) -> tuple[Any, Any]:
    position = obs[..., 0].unsqueeze(-1)
    velocity = obs[..., 1].unsqueeze(-1)

    state = dict(
        position=position,
        velocity=velocity,
    )

    target = dict(
        position=torch.ones_like(position) * 0.45,
        velocity=torch.ones_like(velocity) * 5.0,
    )

    return state, target


# Create system
system = MountainCar()

# Create environment
env = make_vec_env(
    "MountainCarContinuous-v0",
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
    std=0.5,
    device="cpu",
)


observation = env.reset()

observation = torch.Tensor(observation.copy())

state, target = obs_to_state_target(observation)


while True:
    action, cost_value = mpc(state, target)

    action_ = action.clone().detach().numpy()

    action_selected = action_[:, 0]
    print(action_selected)
    observation, reward, _, information = env.step(action_selected)
    print(f"Reward: {reward}")
    observation = torch.Tensor(observation.copy())

    state, target = obs_to_state_target(observation)
    print(state)
    env.render("human")
