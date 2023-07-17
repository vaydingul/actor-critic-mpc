import torch
from torch import nn
import pygame
import utils

pygame.font.init()
torch.autograd.set_detect_anomaly(True)


class ModelPredictiveControl(nn.Module):
	def __init__(
		self,
		system,
		action_size=2,
		control_horizon=1,
		prediction_horizon=10,
		num_optimization_step=40,
		lr=1e-2,
		size=10,
		window_size=512,
		agent_location_noise_level=0.05,
		agent_velocity_noise_level=0.05,
		target_location_noise_level=0.05,
		target_velocity_noise_level=0.05,
		force_penalty_level=0.0,
	) -> None:
		super().__init__()
		self.system = system
		self.action_size = action_size
		self.control_horizon = control_horizon
		self.prediction_horizon = prediction_horizon
		self.num_optimization_step = num_optimization_step
		self.lr = lr
		self.size = size
		self.window_size = window_size
		self.agent_location_noise_level = agent_location_noise_level
		self.agent_velocity_noise_level = agent_velocity_noise_level
		self.target_location_noise_level = target_location_noise_level
		self.target_velocity_noise_level = target_velocity_noise_level

	def forward(
		self,
		agent_location: torch.Tensor,
		agent_velocity: torch.Tensor,
		target_location: torch.Tensor,
		target_velocity: torch.Tensor,
	) -> torch.Tensor:
		"""
		Args:
																																																																																																																																		observation (torch.Tensor): The observation tensor. The shape of the tensor is (batch_size, observation_size).
		Returns:
																																																																																																																																		torch.Tensor: The action tensor. The shape of the tensor is (batch_size, action_size).
		"""
		self._agent_location_original = agent_location
		self._agent_velocity_original = agent_velocity
		self._target_location_original = target_location
		self._target_velocity_original = target_velocity

		# Add noise to the observation
		self._agent_location = agent_location + (
			torch.randn_like(agent_location) * self.agent_location_noise_level
		)
		self._agent_velocity = agent_velocity + (
			torch.randn_like(agent_velocity) * self.agent_velocity_noise_level
		)
		self._target_location = target_location + (
			torch.randn_like(target_location) * self.target_location_noise_level
		)
		self._target_velocity = target_velocity + (
			torch.randn_like(target_velocity) * self.target_velocity_noise_level
		)

		return self._optimize(
			agent_location=self._agent_location,
			agent_velocity=self._agent_velocity,
			target_location=self._target_location,
			target_velocity=self._target_velocity,
		)

	def _optimize(
		self, agent_location, agent_velocity, target_location, target_velocity
	) -> None:
		"""
		Optimizes the model.
		"""
		for _ in range(self.num_optimization_step):
			self.optimizer.zero_grad()

			(
				predicted_agent_location,
				predicted_agent_velocity,
				predicted_target_location,
				predicted_target_velocity,
			) = self._predict(
				agent_location, agent_velocity, target_location, target_velocity
			)
			loss = self._loss(
				predicted_agent_location,
				predicted_agent_velocity,
				predicted_target_location,
				predicted_target_velocity,
				# self._target_location_original,
				# self._target_velocity_original,
			)
			self.loss_value = loss
			loss.backward(retain_graph=True)
			self.optimizer.step()

		self._predicted_agent_location = predicted_agent_location.detach()
		self._predicted_agent_velocity = predicted_agent_velocity.detach()
		self._predicted_target_location = predicted_target_location.detach()
		self._predicted_target_velocity = predicted_target_velocity.detach()

		action = self.action#.detach()

		return action

	def _predict(
		self, agent_location, agent_velocity, target_location, target_velocity
	) -> torch.Tensor:
		predicted_agent_location = torch.zeros((self.prediction_horizon, 2))
		predicted_agent_velocity = torch.zeros((self.prediction_horizon, 2))
		predicted_target_location = torch.zeros((self.prediction_horizon, 2))
		predicted_target_velocity = torch.zeros((self.prediction_horizon, 2))

		for i in range(self.prediction_horizon):
			(
				agent_location,
				agent_velocity,
				target_location,
				target_velocity,
			) = self.system(
				agent_location,
				agent_velocity,
				target_location,
				target_velocity,
				self.action[i],
			)
			predicted_agent_location[i] = agent_location
			predicted_agent_velocity[i] = agent_velocity
			predicted_target_location[i] = target_location
			predicted_target_velocity[i] = target_velocity

		return (
			predicted_agent_location,
			predicted_agent_velocity,
			predicted_target_location,
			predicted_target_velocity,
		)

	def _loss(
		self,
		agent_location,
		agent_velocity,
		target_location,
		target_velocity,
	) -> torch.Tensor:
		# Calculate the loss
		# predicted_state -> (prediction_horizon, 4)
		# target -> (4,)

		# Calculate the distance
		loss = torch.norm(agent_location - target_location, 2, -1).mean()

		return loss

	def reset(self) -> None:
		"""
		Resets the model.
		"""
		self.action = torch.zeros((self.prediction_horizon, self.action_size))
		self.action.requires_grad = True
		self.optimizer = torch.optim.Adam([self.action], lr=self.lr)
		self.optimizer.zero_grad()

	def render(self):
		# If not initialized yet, initialize the window
		if not hasattr(self, "window"):
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode((self.window_size, self.window_size))
			self.font = pygame.font.SysFont("Helvetica", 30)
			self.clock = pygame.time.Clock()

		canvas = utils.render_frame(
			size=self.size,
			window_size=self.window_size,
			agent_location_original=self._agent_location_original.clone()
			.detach()
			.numpy(),
			agent_velocity_original=self._agent_velocity_original.clone()
			.detach()
			.numpy(),
			target_location_original=self._target_location_original.clone()
			.detach()
			.numpy(),
			target_velocity_original=self._target_velocity_original.clone()
			.detach()
			.numpy(),
			action=self.action[0].clone().detach().numpy(),
			distance_threshold=0.0,
			system=self.system,
			agent_location_noisy=self._agent_location.clone().detach().numpy(),
			agent_velocity_noisy=self._agent_velocity.clone().detach().numpy(),
			target_location_noisy=self._target_location.clone().detach().numpy(),
			target_velocity_noisy=self._target_velocity.clone().detach().numpy(),
			predicted_agent_location=self._predicted_agent_location.clone()
			.detach()
			.numpy(),
			predicted_agent_velocity=self._predicted_agent_velocity.clone()
			.detach()
			.numpy(),
			predicted_target_location=self._predicted_target_location.clone()
			.detach()
			.numpy(),
			predicted_target_velocity=self._predicted_target_velocity.clone()
			.detach()
			.numpy(),
		)

		self.window.blit(canvas, (0, 0))
		pygame.display.flip()
		self.clock.tick(4.0)
