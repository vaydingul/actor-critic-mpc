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
