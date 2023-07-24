#!/usr/bin/env bash

name="models_new/ppo+mpc|noise|no_wind|10|10"

python train_acmpc_multienv_dynamical_system_args.py \
	--n_envs=40 \
	--size=20 \
	--batch_size=2048 \
	--device="cuda" \
	--agent_location_noise_level=0.1 \
	--agent_velocity_noise_level=0.01 \
	--target_location_noise_level=0.1 \
	--target_velocity_noise_level=0.01 \
	--dt=0.1 \
	--random_force_probability=0.0 \
	--random_force_magnitude=0.0 \
	--friction_coefficient=0.25 \
	--wind_gust_x=0.0 \
	--wind_gust_y=0.0 \
	--wind_gust_region_x_min=0.3 \
	--wind_gust_region_x_max=0.7 \
	--wind_gust_region_y_min=0.3 \
	--wind_gust_region_y_max=0.7 \
	--action_size=2 \
	--prediction_horizon=10 \
	--num_optimization_step=10 \
	--lr=2.0 \
	--distance_threshold=1.0 \
	--predict_action=True \
	--predict_cost=False \
	--num_cost_terms=2 \
	--total_timesteps=1000000 \
	--tb_log_name="$name" \
	--save_name="$name"
