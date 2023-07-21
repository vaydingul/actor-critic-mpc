import numpy as np
from gymnasium import ObservationWrapper

import gymnasium as gym
from gymnasium.spaces import Box, Dict


class RelativeRedundant(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = Dict(
            {
                "relative_location": Box(
                    low=-self.size, high=self.size, shape=(2,), dtype=np.float32
                ),
                "relative_velocity": Box(
                    low=-10, high=10, shape=(2,), dtype=np.float32
                ),
                "agent_location": Box(
                    low=0, high=self.size, shape=(2,), dtype=np.float32
                ),
                "agent_velocity": Box(low=-10, high=10, shape=(2,), dtype=np.float32),
                "target_location": Box(
                    low=0, high=self.size, shape=(2,), dtype=np.float32
                ),
                "target_velocity": Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            }
        )

    def observation(self, observation) -> np.ndarray:
        return {
            "relative_location": observation["agent_location"]
            - observation["target_location"],
            "relative_velocity": observation["agent_velocity"]
            - observation["target_velocity"],
            "agent_location": observation["agent_location"],
            "agent_velocity": observation["agent_velocity"],
            "target_location": observation["target_location"],
            "target_velocity": observation["target_velocity"],
        }


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
