import numpy as np
import gymnasium as gym
from typing import Tuple, Optional, Any, List, Union
from copy import deepcopy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import torch
from torch import nn


class StateSpaceModel(nn.Module):
	def __init__(
		self,
		a_matrix: np.ndarray,
		b_matrix: np.ndarray,
	):
		super(StateSpaceModel, self).__init__()

		self.a_matrix = torch.from_numpy(a_matrix)
		self.b_matrix = torch.from_numpy(b_matrix)

	def forward(
		self,
		state: torch.Tensor,
		action: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		state (torch.Tensor): Observation from the environment.
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		action (torch.Tensor): Action taken in the environment.
		Returns:
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		Observation from the environment.


		"""
		# return self.a_matrix @ state + self.b_matrix @ action
		return torch.bmm(self.a_matrix, state) + torch.bmm(self.b_matrix, action)

class GymEnvironmentLinearizer:
	def __init__(
		self,
		env: Union[gym.Env, VecEnv],
		eps: Union[List[float], float] = 1e-1,
		state_dynamics: bool = True,
		reward_dynamics: bool = False,
	):
		assert (
			state_dynamics != reward_dynamics
		), "At least one of state_dynamics and reward_dynamics must be True"
		assert isinstance(env, gym.Env) or isinstance(
			env, VecEnv
		), "env must be a gym.Env or VecEnv!"

		self.env = env
		self.eps = eps
		self.state_dynamics = state_dynamics
		self.reward_dynamics = reward_dynamics

		# Fetch observation space and action space
		self.observation_space = env.observation_space
		self.action_space = env.action_space

		self.flatten_observation = False
		# Check spaces
		# If observation space is not 1D, set a flag to flatten it
		if len(self.observation_space.shape) > 1:
			self.flatten_observation = True

		# Calculate the size of the flattened observation space
		self.observation_shape = self.observation_space.shape
		self.observation_size = np.prod(self.observation_space.shape)
		self.action_size = np.prod(self.action_space.shape)

	def linearize(
		self, observation: np.ndarray, action: np.ndarray = None
	) -> StateSpaceModel:
		if action is None:
			action = np.zeros_like(self.action_space.sample())

		# Flatten observation if necessary
		if self.flatten_observation:
			observation = observation.flatten()

		batch_size = observation.shape[0]

		# Calculate state dynamics
		action_dynamics_matrix = self._calculate_dynamics(observation, action)

		if self.state_dynamics:
			state_dynamics_matrix = np.zeros(
				(batch_size, self.observation_size, self.observation_size),
				dtype=np.float32,
			)
			return StateSpaceModel(
				a_matrix=state_dynamics_matrix,
				b_matrix=action_dynamics_matrix,
			)
		else:
			reward_dynamics_matrix = np.zeros((batch_size, 1, 1))
			return StateSpaceModel(
				a_matrix=reward_dynamics_matrix,
				b_matrix=action_dynamics_matrix,
			)

	def _calculate_dynamics(
		self, observation: np.ndarray, action: np.ndarray
	) -> Tuple[np.ndarray, np.ndarray]:
		batch_size = observation.shape[0]

		if self.state_dynamics:
			index = 0
			size = self.observation_size
		else:
			index = 1
			size = 1

		action_dynamics_matrix = np.zeros(
			(batch_size, size, self.action_size), dtype=np.float32
		)
		for k in range(self.action_size):
			action_ = np.copy(action)
			# Add epsilon to the kth action
			action_[..., k] += self.eps
			# Copy the environment
			env = deepcopy(self.env)

			forward = env.step(action_)[index]

			action_ = np.copy(action)
			# Substract epsilon to the kth action
			action_[..., k] -= self.eps
			# Copy the environment
			env = deepcopy(self.env)

			backward = env.step(action_)[index]

			action_dynamics_matrix[..., k] = (forward - backward) / (2 * self.eps)

		return action_dynamics_matrix

	def __call__(
		self,
		observation: np.ndarray,
		action: np.ndarray = None,
	) -> StateSpaceModel:
		return self.linearize(observation, action)
