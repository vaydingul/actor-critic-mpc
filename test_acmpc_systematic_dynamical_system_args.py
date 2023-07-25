from glob import glob
import env
from argparse import ArgumentParser
from policy import (
    ActorCriticModelPredictiveControlPolicy,
    ActorCriticModelPredictiveControlFeatureExtractor,
)
import gymnasium as gym
import numpy as np
from wrapper import RelativeRedundant
from stable_baselines3 import PPO
from system import DynamicalSystem
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd

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
    device = args.device
    models_path = args.models_path
    evaluation_name = args.evaluation_name
    num_episodes = args.num_episodes

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

    # Environment parameters
    distance_threshold = args.distance_threshold

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

    # Create environment
    env = gym.make(
        "DynamicalSystem-v0",
        render_mode=None,
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

    results_dict = dict()
    results_dict["model"] = ["mean_reward", "std_reward", "mean_length", "std_length"]

    for file in glob(models_path + "*.zip"):


        try:
            # Create model
            model = PPO.load(file, device=device)
            print(f"Loaded model {file}")
        except:
            print(f"Failed to load model {file}")
            continue
        # Evaluate model
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=num_episodes,
            deterministic=False,
            render=False,
            callback=None,
            reward_threshold=None,
            return_episode_rewards=True,
        )
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        results_dict[file] = [mean_reward, std_reward, mean_length, std_length]

    # Save via pandas
    results_df = pd.DataFrame.from_dict(results_dict, orient="index")
    results_df.to_csv(evaluation_name + ".csv")



if __name__ == "__main__":
    argprs = ArgumentParser()
    argprs.add_argument("--size", type=int, default=20)
    argprs.add_argument("--device", type=str, default="cuda")
    argprs.add_argument("--models_path", type=str, default="models_new/")
    argprs.add_argument("--evaluation_name", type=str, default="high_noise_high_wind_models_new")
    argprs.add_argument("--num_episodes", type=int, default=100)
    argprs.add_argument("--agent_location_noise_level", type=float, default=0.5)
    argprs.add_argument("--agent_velocity_noise_level", type=float, default=0.1)
    argprs.add_argument("--target_location_noise_level", type=float, default=0.5)
    argprs.add_argument("--target_velocity_noise_level", type=float, default=0.1)
    argprs.add_argument("--dt", type=float, default=0.1)
    argprs.add_argument("--random_force_probability", type=float, default=0.001)
    argprs.add_argument("--random_force_magnitude", type=float, default=10.0)
    argprs.add_argument("--friction_coefficient", type=float, default=0.25)
    argprs.add_argument("--wind_gust_x", type=float, default=1.0)
    argprs.add_argument("--wind_gust_y", type=float, default=1.0)
    argprs.add_argument("--wind_gust_region_x_min", type=float, default=0.2)
    argprs.add_argument("--wind_gust_region_x_max", type=float, default=0.8)
    argprs.add_argument("--wind_gust_region_y_min", type=float, default=0.2)
    argprs.add_argument("--wind_gust_region_y_max", type=float, default=0.8)
    argprs.add_argument("--distance_threshold", type=float, default=1.0)

    args = argprs.parse_args()

    main(args)
