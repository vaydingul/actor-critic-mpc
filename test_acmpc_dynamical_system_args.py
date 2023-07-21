import env
from argparse import ArgumentParser
from policy import (
    ActorCriticModelPredictiveControlPolicy,
    ActorCriticModelPredictiveControlFeatureExtractor,
)
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from wrapper import RelativePosition, RelativeRedundant
from stable_baselines3 import PPO
from system import DynamicalSystem
from mpc import ModelPredictiveControlSimple, DistributionalModelPredictiveControlSimple

WINDOW_SIZE = 512


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
    batch_size = args.batch_size

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
    )

    # Create environment
    env = gym.make(
        "DynamicalSystem-v0",
        render_mode="human",
        size=size,
        window_size=window_size,
        distance_threshold=distance_threshold,
        system=system,
        agent_location_noise_level=agent_location_noise_level,
        agent_velocity_noise_level=agent_velocity_noise_level,
        target_location_noise_level=target_location_noise_level,
        target_velocity_noise_level=target_velocity_noise_level,
    )
    env = RelativeRedundant(env)

    # Create model
    model = PPO.load("ppo_vanilla")

    while True:
        obs, _ = env.reset()

        done = False
        while not done:
            action, _state = model.predict(obs[np.newaxis], deterministic=True)
            print(action)
            obs, reward, done, _, information = env.step(action.squeeze(0))
            print(done)
            env.render()


if __name__ == "__main__":
    argprs = ArgumentParser()
    argprs.add_argument("--size", type=int, default=10)
    argprs.add_argument("--batch_size", type=int, default=2048)
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
    argprs.add_argument("--tb_log_name", type=str, default="PPO_vanilla_size_10")
    argprs.add_argument("--save_name", type=str, default="PPO_vanilla_size_10")

    args = argprs.parse_args()

    main(args)
