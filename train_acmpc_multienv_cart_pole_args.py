import env
from argparse import ArgumentParser
from policy import (
    ActorCriticModelPredictiveControlPolicy,
)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from system import CartPole
from mpc import ModelPredictiveControlWithoutOptimizer
from typing import Callable, Any

from torch import nn
import torch
import numpy as np

from wandb.integration.sb3 import WandbCallback
import wandb
from wrapper import GaussianNoiseWrapper
from costs import cart_pole_cost, cart_pole_obs_to_state_target

from utils import str_2_bool


def main(args):
    n_envs = args.n_envs
    n_steps = args.n_steps
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    gaussian_noise_scale = args.gaussian_noise_scale

    # System parameters
    goal_velocity = args.goal_velocity
    continuous_reward = args.continuous_reward
    # MPC parameters
    action_size = args.action_size
    prediction_horizon = args.prediction_horizon
    num_optimization_step = args.num_optimization_step
    lr = args.lr

    # Policy parameters
    predict_action = str_2_bool(args.predict_action)
    predict_cost = str_2_bool(args.predict_cost)
    num_cost_terms = args.num_cost_terms

    # Learning parameters
    total_timesteps = args.total_timesteps

    group_name = args.group_name
    log_name = args.log_name
    save_name = args.save_name

    # Create system
    system = CartPole()

    # Create Model Predictive Control model
    mpc_class = ModelPredictiveControlWithoutOptimizer
    mpc_kwargs = dict(
        system=system,
        cost=cart_pole_cost,
        action_size=action_size,
        prediction_horizon=prediction_horizon,
        num_optimization_step=num_optimization_step,
        lr=lr,
        std=0.5,
        device=device,
    )

    env = make_vec_env(
        "CartPoleContinuous-v0",
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(continuous_reward=continuous_reward),
        vec_env_cls=SubprocVecEnv,
    )
    env.seed(seed)

    # Policy
    if num_optimization_step == 0:
        policy_class = "MlpPolicy"
        policy_kwargs = dict()
    else:
        policy_class = ActorCriticModelPredictiveControlPolicy
        policy_kwargs = dict(
            mpc_class=mpc_class,
            mpc_kwargs=mpc_kwargs,
            predict_action=predict_action,
            predict_cost=predict_cost,
            num_cost_terms=num_cost_terms,
            obs_to_state_target=cart_pole_obs_to_state_target,
        )

    # WandB integration
    run = wandb.init(
        project="acmpc",
        group=group_name,
        name=log_name,
        config=args,
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # Create model
    model = PPO(
        policy_class,
        env,
        verbose=2,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        tensorboard_log="tensorboard_logs",
        device=device,
        ent_coef=0.0,
        gae_lambda=0.8,
        gamma=0.98,
        learning_rate=1e-3,
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

    argprs.add_argument("--n_envs", type=int, default=32)
    argprs.add_argument("--n_steps", type=int, default=256)
    argprs.add_argument("--batch_size", type=int, default=32 * 256)
    argprs.add_argument("--device", type=str, default="cpu")
    argprs.add_argument("--seed", type=int, default=42)
    argprs.add_argument("--gaussian_noise_scale", type=float, default=0.0)

    argprs.add_argument("--goal_velocity", type=float, default=0.00)
    argprs.add_argument("--continuous_reward", type=str, default="True")

    argprs.add_argument("--action_size", type=int, default=1)
    argprs.add_argument("--prediction_horizon", type=int, default=4)
    argprs.add_argument("--num_optimization_step", type=int, default=4)
    argprs.add_argument("--lr", type=float, default=1.0)

    argprs.add_argument("--predict_action", type=str, default="True")
    argprs.add_argument("--predict_cost", type=str, default="False")
    argprs.add_argument("--num_cost_terms", type=int, default=2)
    argprs.add_argument("--total_timesteps", type=int, default=1_000_000)

    argprs.add_argument("--group_name", type=str, default="cartpole_continuous_reward")
    argprs.add_argument("--log_name", type=str, default="acmpc_4_4")
    argprs.add_argument("--save_name", type=str, default="m4")

    args = argprs.parse_args()

    main(args)
