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
from mpc import ModelPredictiveControlSimple
from typing import Callable

WINDOW_SIZE = 512


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
    tb_log_folder = args.tb_log_folder
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
    mpc_class = ModelPredictiveControlSimple
    mpc_kwargs = dict(
        system=system,
        action_size=action_size,
        prediction_horizon=prediction_horizon,
        num_optimization_step=num_optimization_step,
        size=size,
        lr=lr,
        device=device,
    )

    # Create environment
    env = make_env(
        rank=0,
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

    env = SubprocVecEnv(
        [
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
    )

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
    argprs.add_argument("--size", type=int, default=10)
    argprs.add_argument("--n_envs", type=int, default=4)
    argprs.add_argument("--n_steps", type=int, default=2048)
    argprs.add_argument("--batch_size", type=int, default=2048)
    argprs.add_argument("--device", type=str, default="cuda")
    argprs.add_argument("--agent_location_noise_level", type=float, default=0.0)
    argprs.add_argument("--agent_velocity_noise_level", type=float, default=0.0)
    argprs.add_argument("--target_location_noise_level", type=float, default=0.0)
    argprs.add_argument("--target_velocity_noise_level", type=float, default=0.0)
    argprs.add_argument("--dt", type=float, default=0.1)
    argprs.add_argument("--random_force_probability", type=float, default=0.0)
    argprs.add_argument("--random_force_magnitude", type=float, default=10.0)
    argprs.add_argument("--friction_coefficient", type=float, default=0.25)
    argprs.add_argument("--wind_gust_x", type=float, default=0.5)
    argprs.add_argument("--wind_gust_y", type=float, default=0.5)
    argprs.add_argument("--wind_gust_region_x_min", type=float, default=0.3)
    argprs.add_argument("--wind_gust_region_x_max", type=float, default=0.7)
    argprs.add_argument("--wind_gust_region_y_min", type=float, default=0.3)
    argprs.add_argument("--wind_gust_region_y_max", type=float, default=0.7)
    argprs.add_argument("--action_size", type=int, default=2)
    argprs.add_argument("--prediction_horizon", type=int, default=2)
    argprs.add_argument("--num_optimization_step", type=int, default=2)
    argprs.add_argument("--lr", type=float, default=2.0)
    argprs.add_argument("--distance_threshold", type=float, default=1.0)
    argprs.add_argument("--predict_action", type=str, default="True")
    argprs.add_argument("--predict_cost", type=str, default="False")
    argprs.add_argument("--num_cost_terms", type=int, default=2)
    argprs.add_argument("--total_timesteps", type=int, default=100_000)
    argprs.add_argument("--tb_log_folder", type=str, default="PPO_vanilla_size_10")
    argprs.add_argument("--tb_log_name", type=str, default="PPO_vanilla_size_10")
    argprs.add_argument("--save_name", type=str, default="PPO_vanilla_size_10")

    args = argprs.parse_args()

    main(args)
