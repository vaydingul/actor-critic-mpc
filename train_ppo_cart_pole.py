import env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env_id = "CartPoleContinuous-v0"
env_ = make_vec_env(env_id, n_envs=1)

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env_,
    batch_size=2048,
    n_steps=2048,
    verbose=1,
)


# Train the agent
model.learn(total_timesteps=int(1e6), progress_bar=True, tb_log_name="pendulum_ppo")

# Fetch model
vec_env = model.get_env()

# Reset environment
obs = vec_env.reset()

while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, information = vec_env.step(action)
    vec_env.render("human")
