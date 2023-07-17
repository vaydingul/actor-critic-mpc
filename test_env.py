import env
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from wrapper import RelativePosition
from stable_baselines3 import A2C, PPO
from system import DynamicalSystem


def main():
	size = 10

	# Create system
	system = DynamicalSystem(size = size,
			  random_force_probability=0.01,
			  wind_gust=[0.1, 0.1],)

	# Create environment
	env = gym.make(
		"DynamicalSystem-v0",
		render_mode="human",
		size=size,
		distance_threshold=0.2,
		system=system,
	)
	env = RelativePosition(env)
	env = FlattenObservation(env)
	env.reset()

	while True:
		action = np.zeros(2)
		obs, reward, done, _, information = env.step(action)
		env.render()


if __name__ == "__main__":
	main()
