import torch
from torch import nn
import pygame
pygame.font.init()


class ModelPredictiveControl(nn.Module):
    
	def __init__(self, system, action_size = 2, control_horizon = 1, prediction_horizon = 10,num_optimization_step = 40, lr = 1e-2, size = 10, window_size = 512) -> None:
		super().__init__()
		self.system = system
		self.action_size = action_size
		self.control_horizon = control_horizon
		self.prediction_horizon = prediction_horizon
		self.num_optimization_step = num_optimization_step
		self.lr = lr
		self.size = size
		self.window_size = window_size
	
		

	def forward(self, observation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			observation (torch.Tensor): The observation tensor. The shape of the tensor is (batch_size, observation_size).
		Returns:
			torch.Tensor: The action tensor. The shape of the tensor is (batch_size, action_size).
		"""
		self._observation = observation
		self._target = target

		return self._optimize(observation, target)

	
	def _optimize(self, observation, target) -> None:
		"""
		Optimizes the model.
		"""
		for _ in range(self.num_optimization_step):

			self.optimizer.zero_grad()

			predicted_state = self._predict(observation)
			loss = self._loss(predicted_state, target)
			self.loss_value = loss
			loss.backward(retain_graph=True)
			self.optimizer.step()

		action = self.action.detach()
		return action

	def _predict(self, observation) -> torch.Tensor:
		
		predicted_state = torch.zeros((self.prediction_horizon, 4))
		for i in range(self.prediction_horizon):
			observation = self.system(observation, self.action[i])
			predicted_state[i] = observation
		return predicted_state
	
	def _loss(self, predicted_state, target) -> torch.Tensor:

		# Calculate the loss
		# predicted_state -> (prediction_horizon, 4)
		# target -> (4,)
		
		# Calculate the distance
		loss = torch.norm(predicted_state - target, 2, dim=1).mean()
		
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
		
		canvas = pygame.Surface((self.window_size, self.window_size))
		canvas.fill((255, 255, 255))


		# Draw the agent
		agent_location = self._observation[:2]
		agent_location = agent_location.clone().detach().numpy()
		agent_location = self._scale_vector(agent_location)
		pygame.draw.circle(canvas, (0, 0, 255), agent_location, self._scale_size(0.5))

		# Draw the target
		target_location = self._target[:2]
		target_location = target_location.clone().detach().numpy()
		target_location = self._scale_vector(target_location)
		pygame.draw.circle(canvas, (255, 0, 0), target_location, self._scale_size(0.5))

		# Draw the predicted trajectory
		predicted_state = self._predict(self._observation)
		predicted_state = predicted_state.clone().detach().numpy()
		for i in range(self.prediction_horizon):
			location = predicted_state[i, :2]
			location = self._scale_vector(location)
			pygame.draw.circle(canvas, (0, 0, 0), location, self._scale_size(0.1))
		
		# Draw the action
		action = self.action.clone().detach().numpy()
		action = self._scale_vector(action[0])
		pygame.draw.line(canvas, (255, 0, 0), agent_location, (agent_location[0] + action[0] * 5, agent_location[1] + action[1] * 5), 1)

		# Draw the text
		text = self.font.render(f"Loss: {self.loss_value.item():.2f}", True, (0, 0, 0))
		canvas.blit(text, (0, 0))

		self.window.blit(canvas, (0, 0))
		pygame.display.flip()
		self.clock.tick(20.0)




		

	def _scale_vector(self, vector):
		return (
			(vector[0] / self.size * self.window_size).astype(int),
			(vector[1] / self.size * self.window_size).astype(int),
		)
	
	def _scale_size(self, size):
		return int(size / self.size * self.window_size)





