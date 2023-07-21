import env

from policy import (
    ActorCriticModelPredictiveControlPolicy,
    ActorCriticModelPredictiveControlFeatureExtractor,
)
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from wrapper import RelativePosition, RelativeRedundant
from stable_baselines3 import PPO
from system import DynamicalSystem
from mpc import ModelPredictiveControlSimple


def main():
    size = 5
    window_size = 512
    agent_location_noise_level = 0.0
    agent_velocity_noise_level = 0.0
    target_location_noise_level = 0.0
    target_velocity_noise_level = 0.00
    batch_size = 100

    # Create system
    system = DynamicalSystem(
        size=size,
        random_force_probability=0.000,
        random_force_magnitude=10.0,
        friction_coefficient=0.25,
        wind_gust=[0.1, 0.1],
        wind_gust_region=[[0.3, 0.7], [0.3, 0.7]],
    )

    # Create Model Predictive Control model
    mpc_class = ModelPredictiveControlSimple
    mpc_kwargs = dict(
        system=system,
        action_size=2,
        prediction_horizon=5,
        size=size,
        lr=0.5,
        agent_location_noise_level=agent_location_noise_level,
        agent_velocity_noise_level=agent_velocity_noise_level,
        target_location_noise_level=target_location_noise_level,
        target_velocity_noise_level=target_velocity_noise_level,
        num_optimization_step=5,
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
    env = RelativeRedundant(env)
    env = FlattenObservation(env)

    # Feature extractor class
    features_extractor_class = ActorCriticModelPredictiveControlFeatureExtractor
    features_extractor_kwargs = dict(input_dim=4, features_dim=64)
    # Policy
    policy_class = ActorCriticModelPredictiveControlPolicy
    policy_kwargs = dict(
        mpc_class=mpc_class,
        mpc_kwargs=mpc_kwargs,
        predict_action=False,
        predict_cost=True,
        num_cost_terms = 2,
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
    )

    # Create model
    model = PPO(
        policy_class,
        env,
        verbose=2,
        policy_kwargs=policy_kwargs,
        n_steps=batch_size,
        batch_size=batch_size,
    )

    # Train model
    model.learn(total_timesteps=10000, progress_bar=True)

    # Fetch model
    vec_env = model.get_env()

    # Reset environment
    obs = vec_env.reset()

    while True:
        action, _state = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, information = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    main()
