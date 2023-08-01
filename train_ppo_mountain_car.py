from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env_id = "MountainCarContinuous-v0"
env = make_vec_env(env_id, n_envs=8)

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    batch_size=8*256,
    # clip_range=0.1,
    # ent_coef=0.00429,
    # gae_lambda=0.9,
    # gamma=0.999,
    # learning_rate=7.77e-05,
    # max_grad_norm=5.0,
    n_steps=256,
    use_sde=True,
    vf_coef=0.19,
    verbose=1,
)


# Train the agent
model.learn(total_timesteps=int(1e7), progress_bar=True)

# Fetch model
vec_env = model.get_env()

# Reset environment
obs = vec_env.reset()

while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, information = vec_env.step(action)
    vec_env.render("human")
