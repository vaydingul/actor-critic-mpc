#!/usr/bin/env bash

name="dynamical_system_experiments/models_29_07_2023/ppo+mpc|no_noise|wind|1|1"

python train_acmpc_multienv_dynamical_system_args.py \
	--n_envs=8 \
	--size=20 \
	--batch_size=256 \
	--device="cpu" \
	--agent_location_noise_level=0.0 \
	--agent_velocity_noise_level=0.00 \
	--target_location_noise_level=0.0 \
	--target_velocity_noise_level=0.00 \
	--dt=0.1 \
	--random_force_probability=0.0 \
	--random_force_magnitude=0.0 \
	--friction_coefficient=0.25 \
	--wind_gust_x=0.5 \
	--wind_gust_y=0.5 \
	--wind_gust_region_x_min=0.3 \
	--wind_gust_region_x_max=0.7 \
	--wind_gust_region_y_min=0.3 \
	--wind_gust_region_y_max=0.7 \
	--action_size=2 \
	--prediction_horizon=1 \
	--num_optimization_step=1 \
	--lr=2.0 \
	--distance_threshold=1.0 \
	--predict_action=True \
	--predict_cost=False \
	--num_cost_terms=2 \
	--total_timesteps=100000 \
	--tb_log_folder="dynamical_system_experiments/tensorboard_logs_29_07_2023/" \
	--tb_log_name="$name" \
	--save_name="$name"