import env

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from wrapper import RelativePosition
from stable_baselines3 import A2C, PPO

print("Create environment")
env = gym.make("DynamicalSystem-v0", render_mode='rgb_array', size = 10, distance_threshold = 0.2)
env = RelativePosition(env)
env = FlattenObservation(env)
print("Create model")
model = PPO("MlpPolicy", env, verbose=2)

print("Train model")
model.learn(total_timesteps=100_000)

print("Fetch model")
vec_env = model.get_env()

print("Reset environment")
obs = vec_env.reset()

print("Predict")
for i in range(10000000):
	print(i)
	action, _state = model.predict(obs, deterministic=True)
	print(action)
	obs, reward, done, information = vec_env.step(action)
	vec_env.render("human")

	