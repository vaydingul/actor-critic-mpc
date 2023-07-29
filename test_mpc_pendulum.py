from typing import Any
import env
from mpc import ModelPredictiveControlWithoutOptimizer
from system import Pendulum, angle_normalize

# Import make_vec_env to allow parallelization
from stable_baselines3.common.env_util import make_vec_env
import torch


def cost(predicted_state, target_state, action=None, cost_dict=None):
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

    # cost += (
    #     (
    #         ((angle_normalize(predicted_theta) - angle_normalize(target_theta)).pow(2))
    #         * cost_dict["theta_weight"]
    #     )
    #     .mean(0)
    #     .sum()
    # )
    # cost += (
    #     (
    #         (predicted_theta_dot - target_theta_dot).pow(2)
    #         * cost_dict["theta_dot_weight"]
    #     )
    #     .mean(0)
    #     .sum()
    # )

    # cost += (action.pow(2) * cost_dict["action_weight"]).mean(0).sum()

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


def obs_to_state_target(obs) -> tuple[Any, Any]:
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


# Create system
system = Pendulum(
    dt=0.05,
    m=1.0,
    g=10.0,
    l=1.0,
)

# Create environment
env = make_vec_env(
    "Pendulum-v1",
    n_envs=100,
    seed=42,
    env_kwargs=dict(
        g=10.0,
    ),
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


observation = env.reset()

observation = torch.Tensor(observation.copy())

state, target = obs_to_state_target(observation)


while True:
    action, cost_value = mpc(state, target)

    print(action)

    action_ = action.clone().detach().numpy()
    action_selected = action_[:, 0]
    print(action_selected.shape)
    print(f"Action: {action_selected}")
    print(f"Cost: {cost_value}")
    observation, reward, _, information = env.step(action_selected)
    print(reward)
    observation = torch.Tensor(observation.copy())

    state, target = obs_to_state_target(observation)


    # time.sleep(0.1)
    env.render("human")
