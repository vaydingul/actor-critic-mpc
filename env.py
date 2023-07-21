from typing import Any
import numpy as np
import pygame

pygame.font.init()
import utils
import gymnasium as gym
from gymnasium import spaces

from gymnasium.envs.registration import register

register(
    id="DynamicalSystem-v0",
    entry_point="env:DynamicalSystemEnvironment",
    max_episode_steps=500,
    kwargs={
        "size": 5,
        "distance_threshold": 0.5,
        "agent_location_noise_level": 0.0,
        "agent_velocity_noise_level": 0.0,
        "target_location_noise_level": 0.0,
        "target_velocity_noise_level": 0.0,
    },
)


class DynamicalSystemEnvironment(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        system=None,
        size=5,
        window_size=512,
        distance_threshold=0.5,
        agent_location_noise_level=0.05,
        agent_velocity_noise_level=0.05,
        target_location_noise_level=0.05,
        target_velocity_noise_level=0.05,
    ):
        assert (
            system is not None
        ), "System cannot be None. Plug a system into the environment."

        self.system = system
        self.size = size  # The size of the square grid
        self.distance_threshold = (
            distance_threshold  # The distance threshold for the target
        )
        self.agent_location_noise_level = (
            agent_location_noise_level  # The noise level for the agent's location
        )
        self.agent_velocity_noise_level = (
            agent_velocity_noise_level  # The noise level for the agent's velocity
        )
        self.target_location_noise_level = (
            target_location_noise_level  # The noise level for the target's location
        )
        self.target_velocity_noise_level = (
            target_velocity_noise_level  # The noise level for the target's velocity
        )

        self.window_size = window_size  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = utils.make_observation_space(size)

        # We have continuous actions, basically a force vector.
        self.action_space = utils.make_action_space()

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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        # Assign a random continuous location for the agent and the target.
        self._agent_location = self.observation_space["agent_location"].sample()
        self._agent_velocity = self.observation_space["agent_velocity"].sample()
        self._target_location = self.observation_space["target_location"].sample()
        self._target_velocity = self.observation_space["target_velocity"].sample()

        self._agent_location_original = self._agent_location.copy()
        self._agent_velocity_original = self._agent_velocity.copy()
        self._target_location_original = self._target_location.copy()
        self._target_velocity_original = self._target_velocity.copy()

        self._action = np.zeros(2)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
        self._action = action

        (
            _agent_location,
            _agent_velocity,
            _target_location,
            _target_velocity,
        ) = self.system(
            self._agent_location,
            self._agent_velocity,
            self._target_location,
            self._target_velocity,
            action,
        )

        # Real states
        self._agent_location_original = _agent_location.copy()
        self._agent_velocity_original = _agent_velocity.copy()
        self._target_location_original = _target_location.copy()
        self._target_velocity_original = _target_velocity.copy()

        # Noisy observations
        self._agent_location = _agent_location + np.random.normal(
            0, self.agent_location_noise_level, 2
        )
        self._agent_velocity = _agent_velocity + np.random.normal(
            0, self.agent_velocity_noise_level, 2
        )
        self._target_location = _target_location + np.random.normal(
            0, self.target_location_noise_level, 2
        )
        self._target_velocity = _target_velocity + np.random.normal(
            0, self.target_velocity_noise_level, 2
        )

        out_of_bounds = False
        location_satisfied = False
        velocity_satisfied = False

        if (
            self._agent_location_original[0] < 0
            or self._agent_location_original[0] > self.size
            or self._agent_location_original[1] < 0
            or self._agent_location_original[1] > self.size
            or self._target_location_original[0] < 0
            or self._target_location_original[0] > self.size
            or self._target_location_original[1] < 0
            or self._target_location_original[1] > self.size
        ):
            out_of_bounds = True

        observation = self._get_obs()
        info = self._get_info()

        distance = info["distance"]

        if distance < self.distance_threshold:
            location_satisfied = True

        if np.linalg.norm(self._agent_velocity - self._target_velocity) < 0.5:
            velocity_satisfied = True

        # The episode terminates when the agent reaches the target.
        if (location_satisfied and velocity_satisfied) or out_of_bounds:
            terminated = True
        else:
            terminated = False

        # reward = (out_of_bounds * -100.0) + (-location_satisfied * distance * 100) - 10.0
        reward = (
            100.0 if (location_satisfied and velocity_satisfied) else 0
        )  # -distance * 10.0
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
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = utils.render_frame(
            size=self.size,
            window_size=self.window_size,
            agent_location_original=self._agent_location_original,
            agent_velocity_original=self._agent_velocity_original,
            target_location_original=self._target_location_original,
            target_velocity_original=self._target_velocity_original,
            action=self._action,
            distance_threshold=self.distance_threshold,
            system=self.system,
            agent_location_noisy=self._agent_location,
            agent_velocity_noisy=self._agent_velocity,
            target_location_noisy=self._target_location,
            target_velocity_noisy=self._target_velocity,
        )

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

    def _get_obs(self):
        return {
            "agent_location": self._agent_location,
            "agent_velocity": self._agent_velocity,
            "target_location": self._target_location,
            "target_velocity": self._target_velocity,
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
