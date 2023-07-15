import numpy as np
from gymnasium import ObservationWrapper

import gymnasium as gym
from gymnasium.spaces import Box, Dict


class RelativePosition(ObservationWrapper):
    
	def __init__(self, env: gym.Env):
		super().__init__(env)
		self.observation_space = Dict({
							"relative_location": Box(low=-self.size, high=self.size, shape=(2,), dtype=np.float32),
							"agent_velocity": Box(low=-1, high=1, shape=(2,), dtype=np.float32)
							})

	def observation(self, observation) -> np.ndarray:
		return {"relative_location": observation["target_location"] - observation["agent_location"],
	  			"agent_velocity": observation["agent_velocity"]}
	

