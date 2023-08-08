from torch import nn
import torch
import numpy as np
import math


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
        device="cuda",
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
        self.wind_gust_region_x_lower = wind_gust_region[0][0] * size
        self.wind_gust_region_x_upper = wind_gust_region[0][1] * size
        self.wind_gust_region_y_lower = wind_gust_region[1][0] * size
        self.wind_gust_region_y_upper = wind_gust_region[1][1] * size
        self.device = device
        self._TORCH = False

    def forward(self, state, action):
        agent_location = state["agent_location"]
        agent_velocity = state["agent_velocity"]
        target_location = state["target_location"]
        target_velocity = state["target_velocity"]

        self._TORCH = isinstance(agent_location, torch.Tensor)

        if self._TORCH:
            self._ZERO_VECTOR = torch.zeros_like(agent_location, device=self.device)
            self.wind_gust = torch.Tensor(self.wind_gust).to(self.device)

        else:
            self._ZERO_VECTOR = np.zeros_like(agent_location)
            if isinstance(self.wind_gust, torch.Tensor):
                self.wind_gust = self.wind_gust.detach().cpu().numpy()
            else:
                self.wind_gust = np.array(self.wind_gust)

        # Agent propagation

        # Apply the wind gust
        _force_agent = (
            self._ZERO_VECTOR.clone() if self._TORCH else self._ZERO_VECTOR.copy()
        )

        agent_location_x = agent_location[..., 0]
        agent_location_y = agent_location[..., 1]
        agent_location_logical_x = (
            agent_location_x >= self.wind_gust_region_x_lower
        ) * (agent_location_x <= self.wind_gust_region_x_upper)
        agent_location_logical_y = (
            agent_location_y >= self.wind_gust_region_y_lower
        ) * (agent_location_y <= self.wind_gust_region_y_upper)
        agent_location_logical = agent_location_logical_x * agent_location_logical_y

        _force_agent += self.wind_gust * agent_location_logical[..., None]

        _force_agent -= self.friction_coefficient * self._normalize(
            agent_velocity, 2, 1e-6
        )  # Apply friction

        _force_agent += action  # Apply the action

        _acceleration = _force_agent  # Assume mass = 1
        _agent_velocity = agent_velocity + _acceleration * self.dt
        _agent_location = agent_location + agent_velocity * self.dt

        # Target propagation

        # Apply the wind gust
        _force_target = (
            self._ZERO_VECTOR.clone() if self._TORCH else self._ZERO_VECTOR.copy()
        )
        target_location_x = target_location[..., 0]
        target_location_y = target_location[..., 1]
        target_location_logical_x = (
            target_location_x >= self.wind_gust_region_x_lower
        ) * (target_location_x <= self.wind_gust_region_x_upper)
        target_location_logical_y = (
            target_location_y >= self.wind_gust_region_y_lower
        ) * (target_location_y <= self.wind_gust_region_y_upper)
        target_location_logical = target_location_logical_x * target_location_logical_y

        _force_target += self.wind_gust * target_location_logical[..., None]

        # Apply a random force to the target

        if np.random.uniform() < self.random_force_probability:
            if self._TORCH:
                _force_target += torch.Tensor(
                    np.random.uniform(
                        -self.random_force_magnitude, self.random_force_magnitude, 2
                    ),
                ).to(self.device)

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

        # Return the new state
        next_state = dict(
            agent_location=_agent_location,
            agent_velocity=_agent_velocity,
            target_location=_target_location,
            target_velocity=_target_velocity,
        )

        return next_state

    def _normalize(self, vector, norm=2, eps=1e-12):
        if self._TORCH:
            return torch.nn.functional.normalize(vector, norm, -1, eps)
        else:
            return vector / (np.linalg.norm(vector, norm, -1, True) + eps)


class Pendulum(nn.Module):
    def __init__(
        self, dt: float = 0.05, m: float = 1.0, l: float = 1.0, g: float = 10.0
    ) -> None:
        super(Pendulum, self).__init__()

        self.dt = dt
        self.m = m
        self.l = l
        self.g = g

        self.max_speed = 8
        self.max_torque = 2.0

        self._TORCH = False

    def forward(self, state, action):
        theta = state["theta"]
        theta_dot = state["theta_dot"]

        self._TORCH = isinstance(theta, torch.Tensor)

        if self._TORCH:
            clip = torch.clip
            sin = torch.sin
        else:
            clip = np.clip
            sin = np.sin

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = clip(action, -self.max_torque, self.max_torque)

        new_theta_dot = (
            theta_dot + (3 * g / (2 * l) * sin(theta) + 3.0 / (m * l**2) * u) * dt
        )
        new_theta_dot = clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * dt

        next_state = dict(theta=new_theta, theta_dot=new_theta_dot)

        return next_state


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class MountainCar(nn.Module):
    def __init__(self, goal_velocity=0.0):
        super(MountainCar, self).__init__()
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gymnasium, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self._TORCH = False

    def forward(self, state, action):
        position = state["position"]
        velocity = state["velocity"]

        self._TORCH = isinstance(position, torch.Tensor)

        if self._TORCH:
            clip = torch.clip
            cos = torch.cos
        else:
            clip = np.clip
            cos = np.cos

        force = clip(action, self.min_action, self.max_action)

        next_velocity = velocity + force * self.power - 0.0025 * cos(3 * position)
        next_velocity = clip(next_velocity, -self.max_speed, self.max_speed)

        next_position = position + next_velocity
        next_position = clip(next_position, self.min_position, self.max_position)

        next_state = dict(position=next_position, velocity=next_velocity)

        return next_state


class CartPole(nn.Module):
    def __init__(self):
        super(CartPole, self).__init__()
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

        self._TORCH = False

    def forward(self, state, action):
        x = state["x"]
        x_dot = state["x_dot"]
        theta = state["theta"]
        theta_dot = state["theta_dot"]

        self._TORCH = isinstance(x, torch.Tensor)

        if self._TORCH:
            clip = torch.clip
            cos = torch.cos
            sin = torch.sin
        else:
            clip = np.clip
            cos = np.cos
            sin = np.sin

        force = clip(action, -self.force_mag, self.force_mag)

        costheta = cos(theta)
        sintheta = sin(theta)

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
            x_next = x + self.tau * x_dot
            x_dot_next = x_dot + self.tau * xacc
            theta_next = theta + self.tau * theta_dot
            theta_dot_next = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot_next = x_dot + self.tau * xacc
            x_next = x + self.tau * x_dot_next
            theta_dot_next = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot_next

        next_state = dict(
            x=x_next, x_dot=x_dot_next, theta=theta_next, theta_dot=theta_dot_next
        )

        return next_state


class Acrobot(nn.Module):
    dt = 0.2

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    AVAIL_TORQUE = [-1.0, 0.0, +1]

    torque_noise_max = 0.0

    SCREEN_DIM = 500

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        super(Acrobot, self).__init__()

        self._TORCH = False

    def forward(self, state, action):
        theta_1 = state["theta_1"]
        theta_2 = state["theta_2"]
        theta_1_dot = state["theta_1_dot"]
        theta_2_dot = state["theta_2_dot"]

        self._TORCH = isinstance(theta_1, torch.Tensor)

        if self._TORCH:
            augment = lambda arr, dim: torch.cat(arr, dim=dim)
        else:
            augment = lambda arr, dim: np.cat(arr, axis=dim)

        torque = action

        state_augmented = augment(
            arr=[theta_1, theta_2, theta_1_dot, theta_2_dot, torque], dim=1
        )

        next_state = self.rk4(self._dsdt, state_augmented, [0, self.dt])

        next_state[..., 0] = self.wrap(next_state[..., 0], -np.pi, np.pi)
        next_state[..., 1] = self.wrap(next_state[..., 1], -np.pi, np.pi)
        next_state[..., 2] = self.bound(
            next_state[..., 2], -self.MAX_VEL_1, self.MAX_VEL_1
        )
        next_state[..., 3] = self.bound(
            next_state[..., 3], -self.MAX_VEL_2, self.MAX_VEL_2
        )

        return dict(
            theta_1=next_state[..., 0].unsqueeze(-1),
            theta_2=next_state[..., 1].unsqueeze(-1),
            theta_1_dot=next_state[..., 2].unsqueeze(-1),
            theta_2_dot=next_state[..., 3].unsqueeze(-1),
        )

    def _dsdt(self, s_augmented):
        if self._TORCH:
            cos = torch.cos
            sin = torch.sin
        else:
            cos = np.cos
            sin = np.sin

        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[..., -1]
        s = s_augmented[..., :-1]
        theta1 = s[..., 0]
        theta2 = s[..., 1]
        dtheta1 = s[..., 2]
        dtheta2 = s[..., 3]
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - np.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - np.pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    def wrap(self, x, m, M):
        """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
        truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
        For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

        Args:
            x: a scalar
            m: minimum possible value in range
            M: maximum possible value in range

        Returns:
            x: a scalar, wrapped
        """

        diff = M - m
        x = x - (x > M).float() * diff
        x = x + (x < m).float() * diff
        return x

    def bound(self, x, m, M):
        """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
        have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

        Args:
            x: scalar
            m: The lower bound
            M: The upper bound

        Returns:
            x: scalar, bound between min (m) and Max (M)
        """

        if self._TORCH:
            clamp = torch.clamp_
        else:
            clamp = np.clip

        return clamp(x, m, M)

    def rk4(self, derivs, y0, t):
        """
        Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

        Example for 2D system:

            >>> def derivs(x):
            ...     d1 =  x[0] + 2*x[1]
            ...     d2 =  -3*x[0] + 4*x[1]
            ...     return d1, d2

            >>> dt = 0.0005
            >>> t = np.arange(0.0, 2.0, dt)
            >>> y0 = (1,2)
            >>> yout = rk4(derivs, y0, t)

        Args:
            derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
            y0: initial state vector
            t: sample times

        Returns:
            yout: Runge-Kutta approximation of the ODE
        """

        try:
            batch_size, Ny = y0.shape
        except TypeError:
            yout = torch.zeros(
                (
                    batch_size,
                    len(t),
                ),
                dtype=torch.float32,
            )
        else:
            yout = torch.zeros((batch_size, len(t), Ny), dtype=torch.float32)

        yout[:, 0] = y0

        for i in np.arange(len(t) - 1):
            this = t[i]
            dt = t[i + 1] - this
            dt2 = dt / 2.0
            y0 = yout[:, i]

            k1 = torch.as_tensor(derivs(y0))
            k2 = torch.as_tensor(derivs(y0 + dt2 * k1))
            k3 = torch.as_tensor(derivs(y0 + dt2 * k2))
            k4 = torch.as_tensor(derivs(y0 + dt * k3))
            yout[:, i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        # We only care about the final timestep and we cleave off action value which will be zero
        return yout[:, -1][:4]
