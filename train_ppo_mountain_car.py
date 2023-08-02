from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env_id = "MountainCarContinuous-v0"
env = make_vec_env(env_id, n_envs=1)

# The learning agent and hyperparameters
model = PPO(
    policy="MlpPolicy",
    env=env,
    seed=0,
    batch_size=256,
    ent_coef=0.00429,
    learning_rate=7.77e-05,
    n_epochs=10,
    n_steps=8,
    gae_lambda=0.9,
    gamma=0.9999,
    clip_range=0.1,
    max_grad_norm=5,
    vf_coef=0.19,
    use_sde=True,
    policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
    verbose=1,
)


# Train the agent
model.learn(total_timesteps=int(20000), progress_bar=True)

# Fetch model
vec_env = model.get_env()

# Reset environment
obs = vec_env.reset()

while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, information = vec_env.step(action)
    vec_env.render("human")
