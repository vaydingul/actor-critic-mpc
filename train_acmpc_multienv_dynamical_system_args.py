import env
from argparse import ArgumentParser
from policy import (
    ActorCriticModelPredictiveControlPolicy,
    ActorCriticModelPredictiveControlFeatureExtractor,
)
import gymnasium as gym
from wrapper import RelativeRedundant
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import SubprocVecEnv

from system import DynamicalSystem
from mpc import ModelPredictiveControlWithoutOptimizer
from typing import Callable, Any

from torch import nn
import torch

WINDOW_SIZE = 512


def cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["agent_location"].shape
    device = predicted_state["agent_location"].device

    predicted_agent_location = predicted_state["agent_location"]
    predicted_agent_velocity = predicted_state["agent_velocity"]
    # predicted_target_location = predicted_state["target_location"]
    # predicted_target_velocity = predicted_state["target_velocity"]
    target_agent_location = target_state["agent_location"].unsqueeze(1)
    target_agent_velocity = target_state["agent_velocity"].unsqueeze(1)
    # target_target_location = target_state["target_location"]
    # target_target_velocity = target_state["target_velocity"]

    if cost_dict is None:
        cost_dict = dict(
            location_weight=torch.ones(batch_size, prediction_horizon, device=device),
            velocity_weight=torch.ones(batch_size, prediction_horizon, device=device)
            * 0.1,
            action_first_derivative_weight=torch.zeros(
                batch_size, prediction_horizon - 1, device=device
            ),
            action_second_derivative_weight=torch.zeros(
                batch_size, prediction_horizon - 2, device=device
            ),
        )

    cost = torch.tensor(0.0, device=device)

    # Location cost
    cost += (
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
    if action is not None:
        action_first_derivative = torch.diff(action, dim=1)
        cost += (
            (
                cost_dict["action_first_derivative_weight"]
                * torch.norm(action_first_derivative, p=2, dim=-1)
            )
            .mean(dim=1)
            .sum()
        )

        action_second_derivative = torch.diff(action_first_derivative, dim=1)
        cost += (
            (
                cost_dict["action_second_derivative_weight"]
                * torch.norm(action_second_derivative, p=2, dim=-1)
            )
            .mean(dim=1)
            .sum()
        )

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

    return state, target


def make_env(rank: int, seed: int = 0, *args, **kwargs) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(*args, **kwargs)
        env = RelativeRedundant(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def str_2_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def main(args):
    window_size = WINDOW_SIZE

    size = args.size
    n_envs = args.n_envs
    n_steps = args.n_steps
    batch_size = args.batch_size
    device = args.device

    agent_location_noise_level = args.agent_location_noise_level
    agent_velocity_noise_level = args.agent_velocity_noise_level
    target_location_noise_level = args.target_location_noise_level
    target_velocity_noise_level = args.target_velocity_noise_level

    # System parameters
    dt = args.dt
    random_force_probability = args.random_force_probability
    random_force_magnitude = args.random_force_magnitude
    friction_coefficient = args.friction_coefficient
    wind_gust = [args.wind_gust_x, args.wind_gust_y]
    wind_gust_region = [
        [args.wind_gust_region_x_min, args.wind_gust_region_x_max],
        [args.wind_gust_region_y_min, args.wind_gust_region_y_max],
    ]

    # MPC parameters
    action_size = args.action_size
    prediction_horizon = args.prediction_horizon
    num_optimization_step = args.num_optimization_step
    lr = args.lr

    # Environment parameters
    distance_threshold = args.distance_threshold

    # Policy parameters
    predict_action = str_2_bool(args.predict_action)
    predict_cost = str_2_bool(args.predict_cost)
    num_cost_terms = args.num_cost_terms

    # Learning parameters
    total_timesteps = args.total_timesteps
    tb_log_folder = args.tb_log_folder if args.tb_log_folder != "" else None
    tb_log_name = args.tb_log_name
    save_name = args.save_name

    # Create system
    system = DynamicalSystem(
        dt=dt,
        size=size,
        random_force_probability=random_force_probability,
        random_force_magnitude=random_force_magnitude,
        friction_coefficient=friction_coefficient,
        wind_gust=wind_gust,
        wind_gust_region=wind_gust_region,
        device=device,
    )

    # Create Model Predictive Control model
    mpc_class = ModelPredictiveControlWithoutOptimizer
    mpc_kwargs = dict(
        system=system,
        cost=cost,
        action_size=action_size,
        prediction_horizon=prediction_horizon,
        num_optimization_step=num_optimization_step,
        lr=lr,
        device=device,
    )

    # Create environment
    env_list = [
        make_env(
            rank=i,
            seed=0,
            id="DynamicalSystem-v0",
            render_mode="rgb_array",
            size=size,
            window_size=window_size,
            distance_threshold=distance_threshold,
            system=system,
            agent_location_noise_level=agent_location_noise_level,
            agent_velocity_noise_level=agent_velocity_noise_level,
            target_location_noise_level=target_location_noise_level,
            target_velocity_noise_level=target_velocity_noise_level,
        )
        for i in range(n_envs)
    ]
    env = SubprocVecEnv(env_list) if n_envs > 1 else env_list[0]()

    # Feature extractor class
    features_extractor_class = ActorCriticModelPredictiveControlFeatureExtractor
    features_extractor_kwargs = dict(input_dim=4, features_dim=4)
    # Policy
    policy_class = ActorCriticModelPredictiveControlPolicy
    policy_kwargs = dict(
        mpc_class=mpc_class,
        mpc_kwargs=mpc_kwargs,
        predict_action=predict_action,
        predict_cost=predict_cost,
        num_cost_terms=num_cost_terms,
        obs_to_state_target=obs_to_state_target,
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
    )

    # Create model
    model = PPO(
        policy_class,
        env,
        verbose=2,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log=tb_log_folder,
        device=device,
    )

    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name=tb_log_name,
    )

    # Change device to cpu

    model.save(save_name)


if __name__ == "__main__":
    argprs = ArgumentParser()
    argprs.add_argument("--size", type=int, default=20)
    argprs.add_argument("--n_envs", type=int, default=16)
    argprs.add_argument("--n_steps", type=int, default=128)
    argprs.add_argument("--batch_size", type=int, default=16 * 128)
    argprs.add_argument("--device", type=str, default="cpu")
    argprs.add_argument("--agent_location_noise_level", type=float, default=0.5)
    argprs.add_argument("--agent_velocity_noise_level", type=float, default=0.1)
    argprs.add_argument("--target_location_noise_level", type=float, default=0.5)
    argprs.add_argument("--target_velocity_noise_level", type=float, default=0.1)
    argprs.add_argument("--dt", type=float, default=0.1)
    argprs.add_argument("--random_force_probability", type=float, default=0.0)
    argprs.add_argument("--random_force_magnitude", type=float, default=10.0)
    argprs.add_argument("--friction_coefficient", type=float, default=0.25)
    argprs.add_argument("--wind_gust_x", type=float, default=0.0)
    argprs.add_argument("--wind_gust_y", type=float, default=0.0)
    argprs.add_argument("--wind_gust_region_x_min", type=float, default=0.3)
    argprs.add_argument("--wind_gust_region_x_max", type=float, default=0.7)
    argprs.add_argument("--wind_gust_region_y_min", type=float, default=0.3)
    argprs.add_argument("--wind_gust_region_y_max", type=float, default=0.7)
    argprs.add_argument("--action_size", type=int, default=2)
    argprs.add_argument("--prediction_horizon", type=int, default=10)
    argprs.add_argument("--num_optimization_step", type=int, default=0)
    argprs.add_argument("--lr", type=float, default=2.0)
    argprs.add_argument("--distance_threshold", type=float, default=1.0)
    argprs.add_argument("--predict_action", type=str, default="True")
    argprs.add_argument("--predict_cost", type=str, default="False")
    argprs.add_argument("--num_cost_terms", type=int, default=2)
    argprs.add_argument("--total_timesteps", type=int, default=100_000)
    argprs.add_argument("--tb_log_folder", type=str, default="./")
    argprs.add_argument("--tb_log_name", type=str, default="vanilla")
    argprs.add_argument("--save_name", type=str, default="model")

    args = argprs.parse_args()

    main(args)
