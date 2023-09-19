import env
from argparse import ArgumentParser
from policy import (
    ActorCriticModelPredictiveControlPolicy,
    ActorCriticModelPredictiveControlTeacherForcingPolicy,
)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
import numpy as np
from system import Acrobot
from mpc import ModelPredictiveControlWithoutOptimizer
from wrapper import GaussianNoiseWrapper

from utils import str_2_bool

from costs import acrobot_cost, acrobot_obs_to_state_target
from wandb.integration.sb3 import WandbCallback
import wandb


def main(args):
    n_envs = args.n_envs
    n_steps = args.n_steps
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    gaussian_noise_scale = args.gaussian_noise_scale

    # MPC parameters
    action_size = args.action_size
    prediction_horizon = args.prediction_horizon
    num_optimization_step = args.num_optimization_step
    lr = args.lr

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
    system = Acrobot()

    # Create Model Predictive Control model
    mpc_class = ModelPredictiveControlWithoutOptimizer
    mpc_kwargs = dict(
        system=system,
        cost=acrobot_cost,
        action_size=action_size,
        prediction_horizon=prediction_horizon,
        num_optimization_step=num_optimization_step,
        lr=lr,
        device=device,
    )

    # Create the environment
    env_id = "AcrobotContinuous-v0"
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=GaussianNoiseWrapper,
        wrapper_kwargs=dict(std_diff_ratio=gaussian_noise_scale),
        vec_env_cls=SubprocVecEnv,
    )
    env.seed(seed)

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
            obs_to_state_target=acrobot_obs_to_state_target,
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
        # save_code=False,  # optional
    )

    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % 2000 == 0,
    #     video_length=200,
    # )

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
        gae_lambda=0.94,
        gamma=0.99,
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

    argprs.add_argument("--action_size", type=int, default=1)
    argprs.add_argument("--prediction_horizon", type=int, default=2)
    argprs.add_argument("--num_optimization_step", type=int, default=2)
    argprs.add_argument("--lr", type=float, default=1.0)

    argprs.add_argument("--predict_action", type=str, default="True")
    argprs.add_argument("--predict_cost", type=str, default="False")
    argprs.add_argument("--teacher_forcing", action="store_true")
    argprs.add_argument("--num_cost_terms", type=int, default=0)
    argprs.add_argument("--total_timesteps", type=int, default=1_000_000)

    argprs.add_argument("--group_name", type=str, default="acrobot")
    argprs.add_argument("--log_name", type=str, default="acrobot_acmpc_2_2_noisy")
    argprs.add_argument(
        "--save_name", type=str, default="models/acrobot_acmpc_2_2_noisy"
    )

    args = argprs.parse_args()

    main(args)
