from torch import nn
import torch
import numpy as np


class DynamicalSystem(nn.Module):
    def __init__(
        self,
        dt=0.1,
        size=5,
        random_force_probability=0.0,
        random_force_magnitude=0.05,
        friction_coefficient=0.1,
        wind_gust=[0.0, 0.0],
        wind_gust_region=[[0.25, 0.75], [0.25, 0.75]],
    ):
        super(DynamicalSystem, self).__init__()

        self.dt = dt
        self.size = size  # The size of the square grid
        self.random_force_probability = random_force_probability  # The probability of applying a random force to the target
        self.random_force_magnitude = (
            random_force_magnitude  # The magnitude of the random force
        )
        self.friction_coefficient = friction_coefficient  # The friction coefficient
        self.wind_gust = wind_gust  # The wind gust
        self.wind_gust_region = (
            wind_gust_region  # The region where the wind gust is active
        )
        self._TORCH = False
        self._ZERO_VECTOR = np.zeros(2)

    def forward(
        self, agent_location, agent_velocity, target_location, target_velocity, action
    ):
        self._TORCH = isinstance(agent_location, torch.Tensor)

        if self._TORCH:
            self._ZERO_VECTOR = torch.Tensor(self._ZERO_VECTOR)
            self.wind_gust = torch.Tensor(self.wind_gust)

        # Agent propagation

        # Apply the wind gust
        _force_agent = (
            self._ZERO_VECTOR.clone() if self._TORCH else self._ZERO_VECTOR.copy()
        )
        if (
            (agent_location[0] >= self.wind_gust_region[0][0] * self.size)
            and (agent_location[0] <= self.wind_gust_region[0][1] * self.size)
            and (agent_location[1] >= self.wind_gust_region[1][0] * self.size)
            and (agent_location[1] <= self.wind_gust_region[1][1] * self.size)
        ):
            _force_agent += self.wind_gust

        _force_agent -= self.friction_coefficient * self._normalize(
            agent_velocity, 2, -1
        )  # Apply friction

        _force_agent += action  # Apply the action

        _acceleration = _force_agent  # Assume mass = 1
        _agent_velocity = agent_velocity + _acceleration * self.dt
        _agent_location = agent_location + _agent_velocity * self.dt

        # Target propagation

        # Apply the wind gust
        _force_target = (
            self._ZERO_VECTOR.clone() if self._TORCH else self._ZERO_VECTOR.copy()
        )
        if (
            (target_location[0] >= self.wind_gust_region[0][0] * self.size)
            and (target_location[0] <= self.wind_gust_region[0][1] * self.size)
            and (target_location[1] >= self.wind_gust_region[1][0] * self.size)
            and (target_location[1] <= self.wind_gust_region[1][1] * self.size)
        ):
            _force_target += self.wind_gust

        # Apply a random force to the target

        if np.random.uniform() < self.random_force_probability:
            if self._TORCH:
                _force_target += torch.Tensor(
                    np.random.uniform(
                        -self.random_force_magnitude, self.random_force_magnitude, 2
                    )
                )
                print("FORCE TARGET!!!!!")
                print(_force_target)
            else:
                _force_target += np.random.uniform(
                    -self.random_force_magnitude, self.random_force_magnitude, 2
                )

        _force_target -= self.friction_coefficient * self._normalize(
            target_velocity
        )  # Apply friction

        _acceleration = _force_target  # Assume mass = 1
        _target_velocity = target_velocity + _acceleration * self.dt
        _target_location = target_location + target_velocity * self.dt

        return _agent_location, _agent_velocity, _target_location, _target_velocity

    def _normalize(self, vector, norm=2, eps=1e-12):
        if self._TORCH:
            return torch.nn.functional.normalize(vector, norm, -1, eps)
        else:
            return vector / (np.linalg.norm(vector, norm, -1, True) + eps)
