#!/usr/bin/env bash
folder="pendulum_experiments"
name_main="pendulum"
group_name="pendulum_normal_vs_noisy"

prediction_horizon_list=(1 1 2 3 4 5)                                    # 6 7 8 9 10)
num_optimization_step_list=(0 1 2 3 4 5)                                 # 6 7 8 9 10)
name=("ppo" "acmpc_1_1" "acmpc_2_2" "acmpc_3_3" "acmpc_4_4" "acmpc_5_5") # "acmpc_6_6" "acmpc_7_7" "acmpc_8_8" "acmpc_9_9" "acmpc_10_10")

# For loop from each element in the list
for i in "${!prediction_horizon_list[@]}"; do
	# Get the prediction horizon
	prediction_horizon="${prediction_horizon_list[$i]}"
	# Get the number of optimization steps
	num_optimization_step="${num_optimization_step_list[$i]}"
	# Get the name
	name="${name[$i]}"

	# Print the values
	echo "$prediction_horizon"
	echo "$num_optimization_step"
	echo "$name"
	echo "Seed 1"

	# Run the command
	python train_acmpc_multienv_pendulum_args.py --group_name "$group_name" --seed 0208 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --log_name "${name_main}_${name}" --save_name "models/${name_main}/${name}_1"

	# Print the values
	echo "$prediction_horizon"
	echo "$num_optimization_step"
	echo "$name"
	echo "Seed 2"

	# Run the command
	python train_acmpc_multienv_pendulum_args.py --group_name "$group_name" --seed 0411 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --log_name "${name_main}_${name}" --save_name "models/${name_main}/${name}_2"

	# Print the values
	echo "$prediction_horizon"
	echo "$num_optimization_step"
	echo "$name"
	echo "Seed 1 Noisy"

	# Run the command
	python train_acmpc_multienv_pendulum_args.py --group_name "$group_name" --seed 0208 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --log_name "${name_main}_${name}_noisy" --save_name "models/${name_main}/${name}_noisy_1" --gaussian_noise_scale 0.1

	# Print the values
	echo "$prediction_horizon"
	echo "$num_optimization_step"
	echo "$name"
	echo "Seed 2 Noisy"

	# Run the command
	python train_acmpc_multienv_pendulum_args.py --group_name "$group_name" --seed 0411 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --log_name "${name_main}_${name}_noisy" --save_name "models/${name_main}/${name}_noisy_2" --gaussian_noise_scale 0.1

done
