#!/usr/bin/env bash
folder="pendulum_experiments"
name_main="pendulum"

prediction_horizon_list=(6 7)    #(1 2 5 8 10)
num_optimization_step_list=(6 7) #(0 2 5 8 10)
name=("acmpc_6_6" "acmpc_7_7")   #("ppo" "acmpc_2_2" "acmpc_5_5" "acmpc_8_8" "acmpc_10_10")

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
	python train_acmpc_multienv_pendulum_args.py --seed 42 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --tb_log_folder "$folder/tensorboard_logs_new_policy/" --tb_log_name "${name_main}_${name}" --save_name "$folder/models_new_policy/${name_main}_${name}_1"

	# Print the values
	echo "$prediction_horizon"
	echo "$num_optimization_step"
	echo "$name"
	echo "Seed 2"

	# Run the command
	python train_acmpc_multienv_pendulum_args.py --seed 2023 --prediction_horizon "$prediction_horizon" --num_optimization_step "$num_optimization_step" --tb_log_folder "$folder/tensorboard_logs_new_policy/" --tb_log_name "${name_main}_${name}" --save_name "$folder/models_new_policy/${name_main}_${name}_2"
done
