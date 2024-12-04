import pandas as pd
import numpy as np





grid = pd.read_csv('map_environment/map_files/grid_world_3.csv')
start_state = 0
goal_state = 99
max_iters = 1000
revisit_penalty = 0.1
bias_factor = 0.1
obstacle_weight = 10
shuffle = True
states, actions_opt = mpc_loop(start_state, max_iters, grid, goal_state, revisit_penalty, bias_factor, obstacle_weight, shuffle)

