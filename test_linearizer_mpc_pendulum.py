import time
from typing import Any
import env
from mpc import EnvironmentPredictiveControlWithoutOptimizer
from system import Pendulum, angle_normalize

# Import make_vec_env to allow parallelization
from stable_baselines3.common.env_util import make_vec_env
import torch

from copy import deepcopy


def cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state.shape
    device = predicted_state.device

    if cost_dict is None:
        cost_dict = dict(
            state_weight=torch.ones_like(predicted_state, device=device) * 0.5,
            action_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.01,
        )

    cost = (
        (
            torch.nn.functional.mse_loss(
                predicted_state,
                target_state,
                reduction="none",
            )
            * cost_dict["state_weight"]
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
    state = obs

    target = torch.zeros_like(state)
    target[..., 0] = 1.0

    return state, target


# Create environment
env = make_vec_env(
    "Pendulum-v1",
    n_envs=1,
    seed=42,
    env_kwargs=dict(
        g=10.0,
    ),
)

env_render = deepcopy(env)

# Create Model Predictive Control model
mpc = EnvironmentPredictiveControlWithoutOptimizer(
    env,
    cost,
    action_size=1,
    prediction_horizon=1,
    num_optimization_step=50,
    lr=0.1,
    std=0.2,
    device="cpu",
)

env.seed(42)
env_render.seed(42)
observation = env.reset()
env_render.reset()

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
    env_render.step(action_selected)

    # print(reward)
    observation = torch.Tensor(observation.copy())

    state, target = obs_to_state_target(observation)

    env_render.render("human")
