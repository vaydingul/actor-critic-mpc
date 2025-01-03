from linearizer import GymEnvironmentLinearizer
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from copy import deepcopy


def main():
    env_id = "Pendulum-v1"
    env = make_vec_env(env_id, n_envs=1)

    linearizer = GymEnvironmentLinearizer(env=env, eps=0.1, state_dynamics=True)

    obs = env.reset()

    for k in range(100000):
        # Sample an action
        action = np.abs(env.action_space.sample().reshape(1, -1))

        # Linearize around the current state and action
        delta_state_dynamics = linearizer(obs, action)

        delta_state = torch.from_numpy(np.zeros_like(obs, dtype=np.float32)).unsqueeze(
            -1
        )

        delta_action_numpy = np.zeros_like(action, dtype=np.float32)
        delta_action_numpy[..., 0] += 0.5

        delta_action = torch.from_numpy(delta_action_numpy).unsqueeze(-1)

        # Compute the next state
        next_state_predicted = (
            delta_state_dynamics(delta_state, delta_action).squeeze(-1).numpy()
        )

        # Compute the next state
        next_obs, reward, done, information = env.step(action + delta_action_numpy)

        # Compute the error
        error = next_obs - next_state_predicted

        # print(f"Delta state dynamics A matrix: {delta_state_dynamics.a_matrix}")
        # print(f"Delta state dynamics B matrix: {delta_state_dynamics.b_matrix}")
        # print(f"Action: {action}")
        # print(f"Delta action: {delta_action}")
        # print(f"Next action: {action + delta_action_numpy}")

        print(f"Error: {error.sum()}")
        # print(f"State: {obs}")
        # print(f"Next state: {next_obs}")
        # print(f"Predicted next delta state: {next_state_predicted}")

        obs = next_obs



if __name__ == "__main__":
    main()
