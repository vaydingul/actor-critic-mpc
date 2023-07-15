from typing import Any
import numpy as np
import pygame
pygame.font.init()

import gymnasium as gym
from gymnasium import spaces

from gymnasium.envs.registration import register

register(
	id="DynamicalSystem-v0",
	entry_point="env:DynamicalSystemEnvironment",
	max_episode_steps=500,
	kwargs={"size": 5, "distance_threshold": 0.5},
)

class DynamicalSystemEnvironment(gym.Env):
	
	metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

	def __init__(self, render_mode=None, size=5, distance_threshold = 0.5):
		self.size = size  # The size of the square grid
		self.distance_threshold = distance_threshold # The distance threshold for the target
		self.window_size = 512  # The size of the PyGame window

		# Observations are dictionaries with the agent's and the target's location.
		# Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
		self.observation_space = spaces.Dict(
			{
				"agent_location": spaces.Box(0, size, shape=(2,), dtype=float),
				"agent_velocity": spaces.Box(-10, 10, shape=(2,), dtype=float), # (-inf, inf)
				"target_location": spaces.Box(0, size, shape=(2,), dtype=float),
			}
		)
		
		# We have continuous actions, basically a force vector.
		self.action_space = spaces.Box(-10.0, 10.0, shape=(2,), dtype=float)

		self._dt = 0.1  # The time step of the simulation
		
		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode

		"""
		If human-rendering is used, `self.window` will be a reference
		to the window that we draw to. `self.clock` will be a clock that is used
		to ensure that the environment is rendered at the correct framerate in
		human-mode. They will remain `None` until human-mode is used for the
		first time.
		"""
		self.window = None
		self.clock = None


	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
		super().reset(seed=seed, options=options)

		# Assign a random continuous location for the agent and the target.
		self._agent_location = self.observation_space["agent_location"].sample()
		self._agent_velocity = self.observation_space["agent_velocity"].sample()
		self._target_location = self.observation_space["target_location"].sample()
		self._action = np.zeros(2)
		observation = self._get_obs()
		info = self._get_info()

		if self.render_mode == "human":
			self._render_frame()

		return observation, info

	def step(self, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
		self._action = action
		_force = action - self._get_dir_vec() * 0.1
		_acceleration = _force # Assume mass = 1

		_velocity = self._agent_velocity + _acceleration * self._dt

		# Clip the velocity
		# _velocity = np.clip(_velocity, -1, 1)

		_location = self._agent_location + _velocity * self._dt
		# _location = np.clip(_location, 0, self.size)
		
		out_of_bounds = False
		target_reached = False
		if _location[0] < 0 or _location[0] > self.size or _location[1] < 0 or _location[1] > self.size:
			out_of_bounds = True
			_velocity = np.zeros(2)
			_location = self._agent_location
		
		

		self._agent_location = _location
		self._agent_velocity = _velocity

		observation = self._get_obs()
		info = self._get_info()

		distance = info["distance"]
		if (distance < self.distance_threshold) and (np.linalg.norm(self._agent_velocity) < 0.1):
			target_reached = True
		

		# The episode terminates when the agent reaches the target.
		if target_reached or out_of_bounds:
			terminated = True
		else:
			terminated = False
		
		
		# reward = (out_of_bounds * -100.0) + (-target_reached * distance * 100) - 10.0
		reward = 100.0 if target_reached  else 0 #-distance * 10.0
		reward -= out_of_bounds * 100.0
		if self.render_mode == "human":
			self._render_frame()


		return observation, reward, terminated, False, info

	def render(self, render_mode="human"):
		if render_mode is None:
			self.render_mode = render_mode
		elif self.render_mode == "rgb_array":
			return self._render_frame()

		
		
	def _render_frame(self):

		# Create a PyGame window.
		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode(
				(self.window_size, self.window_size)
			)
		if self.clock is None and self.render_mode == "human":
			self.clock = pygame.time.Clock()

		canvas = pygame.Surface((self.window_size, self.window_size))
		canvas.fill((255, 255, 255))

		# Draw the agent.
		agent_location = self._scale_vector(self._agent_location)
		agent_size = self._scale_size(0.2)
		pygame.draw.circle(
			canvas, (255, 0, 0), agent_location, agent_size, width=0
		)

		# Draw the target.
		target_location = self._scale_vector(self._target_location)
		target_size = self._scale_size(0.2)
		pygame.draw.circle(
			canvas, (0, 0, 255), target_location, target_size, width=0
		)

		# Draw a circle to indicate the distance threshold.
		distance_threshold = self._scale_size(self.distance_threshold)
		pygame.draw.circle(
			canvas, (0, 0, 0), target_location, distance_threshold, width=1
		)


		# Draw the velocity vector. Magenta
		velocity_vector = self._scale_vector(0.2 * self._agent_velocity)
		pygame.draw.line(
			canvas,
			(255, 0, 255),
			agent_location,
			(agent_location[0] + velocity_vector[0], agent_location[1] + velocity_vector[1]),
			width=2,
		)

		# Draw the distance to the target.
		distance = self._get_info()["distance"]#self._scale_size(self._get_info()["distance"])
		pygame.draw.line(
			canvas,
			(0, 0, 0),
			agent_location,
			(target_location[0], target_location[1]),
			width=2,
		)
		# Put the distance in the middle of the line.
		text = pygame.font.SysFont("Helvetica", 20).render(
			f"{distance:.2f}", True, (0, 0, 0)
		)

		text_rect = text.get_rect()
		text_rect.center = (
			(agent_location[0] + target_location[0]) / 2,
			(agent_location[1] + target_location[1]) / 2,
		)
		canvas.blit(text, text_rect)

		# Draw the action vector. Cyan
		action_vector = self._scale_vector(0.2 * self._action)
		pygame.draw.line(
			canvas,
			(0, 255, 255),
			agent_location,
			(agent_location[0] + action_vector[0] * 5, agent_location[1] + action_vector[1] * 5),
			width=4,
		)

		# Put additional information on the screen.
		# Action
		text = pygame.font.SysFont("Helvetica", 20).render(
			f"Action: {self._action}", True, (0, 0, 0)
		)
		canvas.blit(text, (10, 10))

		# Target
		text = pygame.font.SysFont("Helvetica", 20).render(
			f"Target: {self._target_location}", True, (0, 0, 0)
		)
		canvas.blit(text, (10, 30))

		# Agent
		text = pygame.font.SysFont("Helvetica", 20).render(
			f"Agent: {self._agent_location}", True, (0, 0, 0)
		)
		canvas.blit(text, (10, 50))

		# Velocity
		text = pygame.font.SysFont("Helvetica", 20).render(
			f"Velocity: {self._agent_velocity}", True, (0, 0, 0)
		)
		canvas.blit(text, (10, 70))



		
		
		


		# Draw the canvas to the window.
		if self.render_mode == "human":
			self.window.blit(canvas, (0, 0))
			pygame.display.update()
			self.clock.tick(self.metadata["render_fps"])
		elif self.render_mode == "rgb_array":
			image = pygame.surfarray.array3d(canvas)
			image = np.transpose(image, axes=(1, 0, 2))
			return image

	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()


	def _scale_vector(self, vector):
		return (
			int(vector[0] / self.size * self.window_size),
			int(vector[1] / self.size * self.window_size),
		)
	
	def _scale_size(self, size):
		return int(size / self.size * self.window_size)
	

	def _get_obs(self):
		return {
			"agent_location": self._agent_location,
			"agent_velocity": self._agent_velocity,
			"target_location": self._target_location,
		}

	def _get_dir_vec(self):
		# Unit vector in the direction of velocity vector
		velocity = self._agent_velocity
		norm = np.linalg.norm(velocity)
		if norm == 0:
			return velocity
		return velocity / norm
	


	def _get_info(self):
		return {
			"distance": np.linalg.norm(
				self._agent_location - self._target_location, ord=2
			)
		}