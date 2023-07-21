import env

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from wrapper import RelativeRedundant, RelativePosition
from stable_baselines3 import A2C, PPO
from system import DynamicalSystem


def main():
    size = 10
    window_size = 512
    agent_location_noise_level = 0.0
    agent_velocity_noise_level = 0.0
    target_location_noise_level = 0.0
    target_velocity_noise_level = 0.00
    batch_size = 2048

    # Create system
    system = DynamicalSystem(
        size=size,
        random_force_probability=0.005,
        random_force_magnitude=10.0,
        friction_coefficient=0.25,
        wind_gust=[0.5, 0.5],
        wind_gust_region=[[0.3, 0.7], [0.3, 0.7]],
    )

    # Create environment
    env = gym.make(
        "DynamicalSystem-v0",
        render_mode="rgb_array",
        size=size,
        window_size=window_size,
        distance_threshold=1.0,
        system=system,
        agent_location_noise_level=agent_location_noise_level,
        agent_velocity_noise_level=agent_velocity_noise_level,
        target_location_noise_level=target_location_noise_level,
        target_velocity_noise_level=target_velocity_noise_level,
        force_penalty_level=0.0,
    )
    env = RelativePosition(env)
    # env = RelativeRedundant(env)
    # env = FlattenObservation(env)

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        n_steps=batch_size,
        batch_size=batch_size,
        tensorboard_log="tensorboard_logs/",
    )

    # Train model
    model.learn(total_timesteps=100_000, progress_bar=True, tb_log_name="PPO_vanilla_size_10")

    # Fetch model
    vec_env = model.get_env()

    # Reset environment
    obs = vec_env.reset()

    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, information = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    main()
