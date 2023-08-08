import env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env_id = "AcrobotContinuous-v0"
env = make_vec_env(env_id, n_envs=16)

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.0,
    gae_lambda=0.94,
    gamma=0.99,
    batch_size=2048,
    n_steps=128,
    verbose=1,
)


# Train the agent
model.learn(total_timesteps=int(1e6), progress_bar=True)

# Fetch model
vec_env = model.get_env()

# Reset environment
obs = vec_env.reset()

while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, information = vec_env.step(action)
    vec_env.render("human")
