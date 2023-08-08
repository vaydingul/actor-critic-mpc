import env
from argparse import ArgumentParser
from policy import (
    ActorCriticModelPredictiveControlPolicy,
)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecVideoRecorder,
)

from system import Pendulum, angle_normalize
from mpc import ModelPredictiveControlWithoutOptimizer
from typing import Callable, Any


import torch

from wandb.integration.sb3 import WandbCallback
import wandb


def cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["theta_1"].shape
    device = predicted_state["theta_1"].device

    predicted_theta_1 = predicted_state["theta_1"]
    predicted_theta_2 = predicted_state["theta_2"]
    predicted_theta_1_dot = predicted_state["theta_1_dot"]
    predicted_theta_2_dot = predicted_state["theta_2_dot"]

    target_theta_1 = (
        target_state["theta_1"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )
    target_theta_1_dot = (
        target_state["theta_1_dot"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )
    target_theta_2 = (
        target_state["theta_2"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )
    target_theta_2_dot = (
        target_state["theta_2_dot"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )

    if cost_dict is None:
        cost_dict = dict(
            theta_1_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
            theta_1_dot_weight=torch.ones(
                batch_size, prediction_horizon, 1, device=device
            )
            * 0.0,
            theta_2_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
            theta_2_dot_weight=torch.ones(
                batch_size, prediction_horizon, 1, device=device
            )
            * 0.0,
            action_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
        )

    cost = (
        (
            torch.nn.functional.mse_loss(
                angle_normalize(predicted_theta_1),
                angle_normalize(target_theta_1),
                reduction="none",
            )
            * cost_dict["theta_1_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                angle_normalize(predicted_theta_2),
                angle_normalize(target_theta_2),
                reduction="none",
            )
            * cost_dict["theta_2_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                predicted_theta_1_dot,
                target_theta_1_dot,
                reduction="none",
            )
            * cost_dict["theta_1_dot_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                predicted_theta_2_dot,
                target_theta_2_dot,
                reduction="none",
            )
            * cost_dict["theta_2_dot_weight"]
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
    theta_1 = torch.atan2(obs[:, 1], obs[:, 0]).unsqueeze(-1)
    theta_2 = torch.atan2(obs[:, 3], obs[:, 2]).unsqueeze(-1)
    theta_1_dot = obs[:, 4].unsqueeze(-1)
    theta_2_dot = obs[:, 5].unsqueeze(-1)

    state = dict(
        theta_1=theta_1,
        theta_2=theta_2,
        theta_1_dot=theta_1_dot,
        theta_2_dot=theta_2_dot,
    )

    target = dict(
        theta_1=torch.ones_like(theta_1) * np.pi,
        theta_2=torch.ones_like(theta_2) * 0.0,
        theta_1_dot=torch.ones_like(theta_1_dot) * 0.0,
        theta_2_dot=torch.ones_like(theta_2_dot) * 0.0,
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
    n_envs = args.n_envs
    n_steps = args.n_steps
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    # System parameters
    dt = args.dt
    m = args.m
    g = args.g
    l = args.l

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

    log_name = args.log_name
    save_name = args.save_name

    # Create system
    system = Acrobot()

    # Create Model Predictive Control model
    mpc_class = ModelPredictiveControlWithoutOptimizer
    mpc_kwargs = dict(
        system=system,
        cost=cost,
        action_size=action_size,
        prediction_horizon=prediction_horizon,
        num_optimization_step=num_optimization_step,
        lr=lr,
        std=0.6,
        device=device,
    )

    env_list = [
        make_env(
            rank=i,
            seed=seed,
            id="Pendulum-v1",
            render_mode="rgb_array",
            g=g,
        )
        for i in range(n_envs)
    ]
    env = SubprocVecEnv(env_list) if n_envs > 1 else env_list[0]()

    # # Feature extractor class
    # features_extractor_class = ActorCriticModelPredictiveControlFeatureExtractor
    # features_extractor_kwargs = dict(input_dim=4, features_dim=4)

    # Policy
    policy_class = ActorCriticModelPredictiveControlPolicy
    policy_kwargs = dict(
        mpc_class=mpc_class,
        mpc_kwargs=mpc_kwargs,
        predict_action=predict_action,
        predict_cost=predict_cost,
        num_cost_terms=num_cost_terms,
        obs_to_state_target=obs_to_state_target,
        # features_extractor_class=features_extractor_class,
        # features_extractor_kwargs=features_extractor_kwargs,
    )

    # WandB integration
    run = wandb.init(
        project="acmpc",
        group="pendulum_without_sde",
        name=log_name,
        config=args,
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x
    #     % ((100000 // (n_envs * n_steps)) * (n_envs * n_steps))
    #     == 0,
    #     video_length=200,
    # )

    if num_optimization_step == 0:
        policy_class = "MlpPolicy"
        policy_kwargs = dict()
    # Create model
    model = PPO(
        policy_class,
        env,
        verbose=2,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log="tensorboard_logs",
        device=device,
        gamma=0.9,
        learning_rate=1e-3,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        # use_sde=True,
        # sde_sample_freq=4,
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
    argprs.add_argument("--dt", type=float, default=0.05)
    argprs.add_argument("--m", type=float, default=1.0)
    argprs.add_argument("--g", type=float, default=10.0)
    argprs.add_argument("--l", type=float, default=1.0)

    argprs.add_argument("--action_size", type=int, default=1)
    argprs.add_argument("--prediction_horizon", type=int, default=5)
    argprs.add_argument("--num_optimization_step", type=int, default=5)
    argprs.add_argument("--lr", type=float, default=1.0)

    argprs.add_argument("--predict_action", type=str, default="True")
    argprs.add_argument("--predict_cost", type=str, default="False")
    argprs.add_argument("--num_cost_terms", type=int, default=3)
    argprs.add_argument("--total_timesteps", type=int, default=1_000_000)
    argprs.add_argument("--log_name", type=str, default="acmpc_5_5_action")
    argprs.add_argument("--save_name", type=str, default="model_acmpc_5_5")

    args = argprs.parse_args()

    main(args)
