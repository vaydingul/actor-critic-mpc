import math
from typing import Any, Optional, Union, Type, Dict
import numpy as np
import pygame

pygame.font.init()
import utils
import gymnasium as gym
from gymnasium.envs.classic_control import utils as gym_utils
from gymnasium import spaces, logger
from gymnasium.error import DependencyNotInstalled

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

register(
    id="CartPoleContinuous-v0",
    entry_point="env:CartPoleContinuousEnv",
    max_episode_steps=500,
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

        state = dict(
            agent_location=self._agent_location_original,
            agent_velocity=self._agent_velocity_original,
            target_location=self._target_location_original,
            target_velocity=self._target_velocity_original,
        )

        next_state = self.system(
            state,
            action,
        )

        _agent_location = next_state["agent_location"]
        _agent_velocity = next_state["agent_velocity"]
        _target_location = next_state["target_location"]
        _target_velocity = next_state["target_velocity"]

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


class CartPoleContinuousEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            -self.force_mag, self.force_mag, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = gym_utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
