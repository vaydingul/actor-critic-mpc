import env
from argparse import ArgumentParser
from policy import (
    ActorCriticModelPredictiveControlPolicy,
    ActorCriticModelPredictiveControlFeatureExtractor,
    ActorCriticModelPredictiveControlTeacherForcingPolicy,
)
import gymnasium as gym
from wrapper import RelativeRedundant
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env


from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
)

from system import DynamicalSystem
from mpc import ModelPredictiveControlWithoutOptimizer


from costs import dynamical_system_cost, dynamical_system_obs_to_state_target
from wandb.integration.sb3 import WandbCallback
import wandb
from utils import str_2_bool
from wrapper import GaussianNoiseWrapperRelativeRedundant

WINDOW_SIZE = 512


def main(args):
    window_size = WINDOW_SIZE

    size = args.size
    n_envs = args.n_envs
    n_steps = args.n_steps
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    gaussian_noise_scale = args.gaussian_noise_scale

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
    teacher_forcing = args.teacher_forcing
    num_cost_terms = args.num_cost_terms

    # Learning parameters
    total_timesteps = args.total_timesteps

    group_name = args.group_name
    log_name = args.log_name
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
        cost=dynamical_system_cost,
        action_size=action_size,
        prediction_horizon=prediction_horizon,
        num_optimization_step=num_optimization_step,
        lr=lr,
        std=2.5,
        device=device,
    )

    env_id = "DynamicalSystem-v0"
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=GaussianNoiseWrapperRelativeRedundant,
        wrapper_kwargs=dict(std_diff_ratio=gaussian_noise_scale),
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(
            size=size,
            window_size=window_size,
            distance_threshold=distance_threshold,
            system=system,
        ),
    )
    env.seed(seed)

    # Feature extractor class
    features_extractor_class = ActorCriticModelPredictiveControlFeatureExtractor
    features_extractor_kwargs = dict(input_dim=4, features_dim=4)
    # Policy
    if num_optimization_step == 0:
        policy_class = "MlpPolicy"
        policy_kwargs = dict()
    else:
        if not teacher_forcing:
            policy_class = ActorCriticModelPredictiveControlPolicy
        else:
            policy_class = ActorCriticModelPredictiveControlTeacherForcingPolicy
        policy_kwargs = dict(
            mpc_class=mpc_class,
            mpc_kwargs=mpc_kwargs,
            predict_action=predict_action,
            predict_cost=predict_cost,
            num_cost_terms=num_cost_terms,
            obs_to_state_target=dynamical_system_obs_to_state_target,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )

    # WandB integration
    run = wandb.init(
        project="acmpc",
        entity="kuavg",
        group=group_name,
        name=log_name,
        config=args,
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        # save_code=False,  # optional
    )

    # Create model
    model = PPO(
        policy_class,
        env,
        verbose=2,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        device=device,
        tensorboard_log=f"tensorboard_logs/",
        seed=seed,
    )

    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=WandbCallback(
            verbose=2,
            model_save_path=f"{save_name}_{run.id}",
            model_save_freq=total_timesteps // 10,
            gradient_save_freq=total_timesteps // 500,
            log="all",
        ),
    )

    run.finish()


if __name__ == "__main__":
    argprs = ArgumentParser()
    argprs.add_argument("--size", type=int, default=20)
    argprs.add_argument("--n_envs", type=int, default=32)
    argprs.add_argument("--n_steps", type=int, default=256)
    argprs.add_argument("--batch_size", type=int, default=32 * 256)
    argprs.add_argument("--device", type=str, default="cpu")
    argprs.add_argument("--seed", type=int, default=42)
    argprs.add_argument("--gaussian_noise_scale", type=float, default=0.0)

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

    argprs.add_argument("--distance_threshold", type=float, default=0.5)
    argprs.add_argument("--predict_action", type=str, default="True")
    argprs.add_argument("--predict_cost", type=str, default="False")
    argprs.add_argument("--teacher_forcing", action="store_true")
    argprs.add_argument("--num_cost_terms", type=int, default=2)
    argprs.add_argument("--total_timesteps", type=int, default=1_000_000)

    argprs.add_argument("--group_name", type=str, default="vanilla")
    argprs.add_argument("--log_name", type=str, default="vanilla")
    argprs.add_argument("--save_name", type=str, default="model")

    args = argprs.parse_args()

    main(args)
