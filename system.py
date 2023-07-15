from torch import nn
import torch
class DynamicalSystem(nn.Module):
    
	def __init__(self, dt = 0.1):
		super(DynamicalSystem, self).__init__()

		self.dt = dt

	def forward(self, observation, action):

		
		location = observation[:2]
		velocity = observation[2:]

		_force = action - self._get_dir_vec(velocity) * 0.1
		_acceleration = _force # Assume mass = 1
		_velocity = velocity + _acceleration * self.dt
		_location = location + _velocity * self.dt

		predicted_state = torch.cat([_location, _velocity])
		
		return predicted_state
	
	def _get_dir_vec(self, vector):

		# Normalize the vector
		norm = torch.norm(vector, 2)
		if norm == 0:
			return vector
		normalized_vector = vector / norm
		return normalized_vector
