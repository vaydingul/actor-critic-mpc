#!/usr/bin/env bash
folder="pendulum_experiments"
name_main="pendulum"
group_name="pendulum_normal_vs_noisy_teacher_forcing"

prediction_horizon_list=(1 2 3 4 5 3 1 5 1 2 3)
num_optimization_step_list=(1 2 3 4 5 1 3 1 5 5 10)
name=("acmpc_1_1" "acmpc_2_2" "acmpc_3_3" "acmpc_4_4" "acmpc_5_5" "acmpc_3_1", "acmpc_1_3", "acmpc_5_1", "acmpc_1_5", "acmpc_2_5", "acmpc_3_10")


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
	python train_acmpc_multienv_pendulum_args.py --group_name "$group_name" --seed 0208 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --log_name "${name_main}_${name}" --save_name "models/${name_main}/${name}_1"  --teacher_forcing

	# Print the values
	echo "$prediction_horizon"
	echo "$num_optimization_step"
	echo "$name"
	echo "Seed 2"

	# Run the command
	python train_acmpc_multienv_pendulum_args.py --group_name "$group_name" --seed 0411 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --log_name "${name_main}_${name}" --save_name "models/${name_main}/${name}_2"  --teacher_forcing

	# Print the values
	echo "$prediction_horizon"
	echo "$num_optimization_step"
	echo "$name"
	echo "Seed 1 Noisy"

	# Run the command
	python train_acmpc_multienv_pendulum_args.py --group_name "$group_name" --seed 0208 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --log_name "${name_main}_${name}_noisy" --save_name "models/${name_main}/${name}_noisy_1" --gaussian_noise_scale 0.1  --teacher_forcing

	# Print the values
	echo "$prediction_horizon"
	echo "$num_optimization_step"
	echo "$name"
	echo "Seed 2 Noisy"

	# Run the command
	python train_acmpc_multienv_pendulum_args.py --group_name "$group_name" --seed 0411 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --log_name "${name_main}_${name}_noisy" --save_name "models/${name_main}/${name}_noisy_2" --gaussian_noise_scale 0.1  --teacher_forcing

done
