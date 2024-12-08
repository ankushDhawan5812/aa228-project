import pandas as pd
import numpy as np
from mpc_tuning import mpc_loop
import csv
import time


def run_parameter_sweep(output_file_path):
    revisit_penalties = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bias_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] # param 2 sweep - same as #2
    grid_worlds = ['map_environment/map_files/grid_world_1.csv', 'map_environment/map_files/grid_world_2.csv', 'map_environment/map_files/grid_world_3.csv']
    num_simulations = 100 # number of simulations per parameter 

    start_state = 0
    goal_state = 99
    max_iters = 200 # should reach the goal within max_iters steps
    obstacle_weight = 10
    shuffle = True

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['grid', 'revisit_penalty', 'bias_factor', 'avg steps to goal', 'reached goal %', 'hit obstacle %', 'steps to goal list'])

    count = 0
    for i, revisit_penalty in enumerate(revisit_penalties): # loop through param 1
        for j, bias_factor in enumerate(bias_factors): # loop through param 2
            start = time.time()
            for grid_world in grid_worlds:
                grid = pd.read_csv(grid_world)

                print(f'running simulations for parameter combo {count} / {len(bias_factors)*len(revisit_penalties)}, grid world {grid_world}')
                reached_goal_count = 0
                hit_obstacle_count = 0
                steps_to_goal_list = []

                for i in range(num_simulations): 

                    # rollout a single simulation
                    states, actions_opt, reached_goal, hit_obstacle, steps_to_goal = mpc_loop(start_state, max_iters, grid, goal_state, revisit_penalty, bias_factor, obstacle_weight, shuffle)

                    if reached_goal:
                        reached_goal_count += 1
                        steps_to_goal_list.append(steps_to_goal)
                    
                    if hit_obstacle:
                        hit_obstacle_count += 1

                avg_steps_to_goal = np.mean(steps_to_goal_list)
                percent_reached_goal = reached_goal_count/num_simulations
                percent_hit_obstacle = hit_obstacle_count/num_simulations

                with open(output_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([grid_world, revisit_penalty, bias_factor, avg_steps_to_goal, percent_reached_goal, percent_hit_obstacle, steps_to_goal_list])
            end = time.time()
            length = end-start
            print(f"one parameter combo took {length} seconds")
            count += 1


                

output_file_path = 'map_environment/sweep_results_3.csv'
run_parameter_sweep(output_file_path)



# plot the results of the mpc experiments
# z axis: total average # steps to goal over all 3 maps
# x axis: revisit penalty
# y axis: bias factor

## simulation runs
# max_iters = 200
# sweep results_2: 
    # revisit_penalties = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # param 1 sweep
    # bias_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # param 2 sweep
# sweep results_3:
    # revisit_penalties = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # bias_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] # param 2 sweep - same as #2

import pandas as pd

def find_valid_and_optimal_thetas(data_path):
    # Load the CSV data
    data = pd.read_csv(data_path)
    
    # Filter out rows where avg steps to goal is NaN
    data = data.dropna(subset=['avg steps to goal'])
    
    # Calculate the sum of avg steps to goal for each (theta1, theta2) pair across all maps
    grouped = (
        data.groupby(['revisit_penalty', 'bias_factor'])
        .agg(
            total_avg_steps=('avg steps to goal', 'sum'),
            reached_goal_min=('reached goal %', 'min')  # Minimum reached goal % across all maps
        )
        .reset_index()
    )
    
    # Apply constraints: reached goal must be 100% across all maps
    valid_data = grouped[grouped['reached_goal_min'] == 1.0]
    
    # If no valid (theta1, theta2) pair meets the constraints, return None
    if valid_data.empty:
        return None, None, "No (theta1, theta2) pair satisfies the constraints."
    
    # Find the row with the minimum total average steps to goal
    optimal_row = valid_data.loc[valid_data['total_avg_steps'].idxmin()]
    
    # Extract optimal theta1, theta2, and minimum steps
    optimal_theta1 = optimal_row['revisit_penalty']
    optimal_theta2 = optimal_row['bias_factor']
    optimal_steps = optimal_row['total_avg_steps']
    
    return valid_data, (optimal_theta1, optimal_theta2, optimal_steps)

# # Example usage:
# data_path = 'map_environment/sweep_results_2.csv'
# valid_params, optimal_thetas = find_valid_and_optimal_thetas(data_path)

# if valid_params is not None:
#     print("Valid parameter sets that satisfy the constraints:")
#     print(valid_params[['revisit_penalty', 'bias_factor', 'total_avg_steps']])
#     print("\nOptimal thetas:")
#     print(f"(theta1={optimal_thetas[0]}, theta2={optimal_thetas[1]}) with minimum total avg steps: {optimal_thetas[2]}")
# else:
#     print(optimal_thetas)



