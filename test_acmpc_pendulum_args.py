import env
from argparse import ArgumentParser
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from system import Pendulum

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

    device = args.device
    model_name = args.model_name

    # System parameters
    dt = args.dt
    m = args.m
    l = args.l
    g = args.g

    # Create system
    system = Pendulum(
        dt=dt,
        m=m,
        l=l,
        g=g,
    )

    # Create environment
    env = make_vec_env(
        env_id="Pendulum-v1",
        n_envs=16,
    )

    model = PPO.load(
        model_name,
        device=device,
    )

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, _, _, _ = env.step(action)

        env.render("human")


if __name__ == "__main__":
    argprs = ArgumentParser()

    argprs.add_argument("--device", type=str, default="cpu")
    argprs.add_argument(
        "--model_name",
        type=str,
        default="pendulum_experiments/models_100000/pendulum_acmpc_9_9_2.zip",
    )
    argprs.add_argument("--dt", type=float, default=0.05)
    argprs.add_argument("--m", type=float, default=1.0)
    argprs.add_argument("--l", type=float, default=1.0)
    argprs.add_argument("--g", type=float, default=10.0)

    args = argprs.parse_args()

    main(args)
