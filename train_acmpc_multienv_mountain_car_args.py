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

from system import MountainCar
from mpc import ModelPredictiveControlWithoutOptimizer
from typing import Callable, Any

from torch import nn
import torch


def cost(predicted_state, target_state, action=None, cost_dict=None):
    batch_size, prediction_horizon, _ = predicted_state["position"].shape
    device = predicted_state["position"].device

    predicted_position = predicted_state["position"]
    predicted_velocity = predicted_state["velocity"]

    target_position = (
        target_state["position"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )
    target_velocity = (
        target_state["velocity"].unsqueeze(1).expand(-1, prediction_horizon, -1)
    )

    if cost_dict is None:
        cost_dict = dict(
            position_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 1000.0,
            velocity_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.0,
            action_weight=torch.ones(batch_size, prediction_horizon, 1, device=device)
            * 0.000,
        )

    cost = (
        (
            torch.nn.functional.mse_loss(
                predicted_position,
                target_position,
                reduction="none",
            )
            * cost_dict["position_weight"]
        )
        .mean(1)
        .sum()
    )

    cost += (
        (
            torch.nn.functional.mse_loss(
                predicted_velocity,
                target_velocity,
                reduction="none",
            )
            * cost_dict["velocity_weight"]
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
    position = obs[..., 0].unsqueeze(-1)
    velocity = obs[..., 1].unsqueeze(-1)

    state = dict(
        position=position,
        velocity=velocity,
    )

    target = dict(
        position=torch.ones_like(position) * 0.45,
        velocity=torch.ones_like(velocity) * 0.0,
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
    goal_velocity = args.goal_velocity

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
    tb_log_folder = args.tb_log_folder if args.tb_log_folder != "" else None
    tb_log_name = args.tb_log_name
    save_name = args.save_name

    # Create system
    system = MountainCar(goal_velocity=goal_velocity)

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

    env_list = [
        make_env(
            rank=i,
            seed=seed,
            id="MountainCarContinuous-v0",
            render_mode="rgb_array",
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

    argprs.add_argument("--n_envs", type=int, default=8)
    argprs.add_argument("--n_steps", type=int, default=256)
    argprs.add_argument("--batch_size", type=int, default=8 * 256)
    argprs.add_argument("--device", type=str, default="cpu")
    argprs.add_argument("--seed", type=int, default=42)
    argprs.add_argument("--goal_velocity", type=float, default=0.00)

    argprs.add_argument("--action_size", type=int, default=1)
    argprs.add_argument("--prediction_horizon", type=int, default=10)
    argprs.add_argument("--num_optimization_step", type=int, default=10)
    argprs.add_argument("--lr", type=float, default=1.0)

    argprs.add_argument("--predict_action", type=str, default="True")
    argprs.add_argument("--predict_cost", type=str, default="False")
    argprs.add_argument("--num_cost_terms", type=int, default=2)
    argprs.add_argument("--total_timesteps", type=int, default=1_000_000)
    argprs.add_argument("--tb_log_folder", type=str, default="dummy")
    argprs.add_argument("--tb_log_name", type=str, default="pendulum_10_10")
    argprs.add_argument("--save_name", type=str, default="dummy_models/pendulum_10_10")

    args = argprs.parse_args()

    main(args)
