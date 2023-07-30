from typing import Any
import env
from mpc import ModelPredictiveControlWithoutOptimizer
from system import DynamicalSystem
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wrapper import RelativeRedundant
import torch


def cost(predicted_state, target_state, action=None, cost_dict=None):
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
            * 0.0,
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

    # Action first and second derivative cost
    # if action is not None:
    #     action_first_derivative = torch.diff(action, dim=1)
    #     cost += (
    #         (
    #             cost_dict["action_first_derivative_weight"]
    #             * torch.norm(action_first_derivative, p=2, dim=-1)
    #         )
    #         .mean(dim=1)
    #         .sum()
    #     )

    #     action_second_derivative = torch.diff(action_first_derivative, dim=1)
    #     cost += (
    #         (
    #             cost_dict["action_second_derivative_weight"]
    #             * torch.norm(action_second_derivative, p=2, dim=-1)
    #         )
    #         .mean(dim=1)
    #         .sum()
    #     )

    return cost


def obs_to_state_target(obs) -> tuple[Any, Any]:
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

    return (state, target)


size = 10.0
agent_location_noise_level = 0.0
agent_velocity_noise_level = 0.00
target_location_noise_level = 0.0
target_velocity_noise_level = 0.00

# Create system
system = DynamicalSystem(
    size=size,
    random_force_probability=0.00,
    random_force_magnitude=10.0,
    friction_coefficient=0.25,
    wind_gust=[0.0, 0.0],
    wind_gust_region=[[0.4, 0.6], [0.4, 0.6]],
    device="cpu",
)

# Create environment
env = make_vec_env(
    "DynamicalSystem-v0",
    n_envs=4,
    seed=0,
    env_kwargs=dict(
        size=size,
        distance_threshold=1.0,
        system=system,
        agent_location_noise_level=agent_location_noise_level,
        agent_velocity_noise_level=agent_velocity_noise_level,
        target_location_noise_level=target_location_noise_level,
        target_velocity_noise_level=target_velocity_noise_level,
    ),
    wrapper_class=RelativeRedundant,
    vec_env_cls=DummyVecEnv,
)


# Create Model Predictive Control model
mpc = ModelPredictiveControlWithoutOptimizer(
    system=system,
    cost=cost,
    action_size=2,
    prediction_horizon=10,
    num_optimization_step=40,
    lr=2.0,
    device="cpu",
)


observation = env.reset()

observation = torch.Tensor(observation)

state, target = obs_to_state_target(observation)

while True:
    (action, cost_value) = mpc(
        state,
        target,
    )

    action_ = action.clone().detach().numpy()
    action_selected = action_[:, 0]

    observation, reward, _, _ = env.step(action_selected)

    observation = torch.Tensor(observation)

    state, target = obs_to_state_target(observation)

    env.render("human")
