import numpy as np
from gymnasium import ObservationWrapper

import gymnasium as gym
from gymnasium.spaces import Box, Dict


class RelativeRedundant(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = Box(
            low=-self.size, high=self.size, shape=(12,), dtype=np.float32
        )

    def observation(self, observation) -> np.ndarray:
        return np.concatenate(
            [
                observation["agent_location"] - observation["target_location"],
                observation["agent_velocity"] - observation["target_velocity"],
                observation["agent_location"],
                observation["agent_velocity"],
                observation["target_location"],
                observation["target_velocity"],
            ]
        )


class RelativePosition(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = Box(
            low=-self.size, high=self.size, shape=(4,), dtype=np.float32
        )

    def observation(self, observation) -> np.ndarray:
        return np.concatenate(
            [
                observation["agent_location"] - observation["target_location"],
                observation["agent_velocity"] - observation["target_velocity"],
            ]
        )


class ExtractRelative(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = Box(
            low=-self.size, high=self.size, shape=(4,), dtype=np.float32
        )

    def observation(self, observation) -> np.ndarray:
        return observation[:4]


class GaussianNoiseWrapper(ObservationWrapper):
    """
    Inherit the existent observation space and add Gaussian noise to it.
    """

    def __init__(self, env: gym.Env, std_diff_ratio: float = 0.1):
        super().__init__(env)

        # Determine mu based on the low and high values of the observation space
        self.mu = (self.observation_space.high + self.observation_space.low) / 2.0
        # Determine std based on the difference between low and high values of the observation space
        self.std = (
            self.observation_space.high - self.observation_space.low
        ) * std_diff_ratio

    def observation(self, observation) -> np.ndarray:
        return observation + np.random.normal(self.mu, self.std, size=observation.shape)
