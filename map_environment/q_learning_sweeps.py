import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sample_trajectories import index_to_state
from sample_trajectories import update_trajectories 
import csv

def save_policy(policy, file_path):
    # print(type(policy))
    with open(file_path, "w") as f:
        for si in range(100):
            f.write(f"{policy[si]}\n")


class QLearning:
    def __init__(self, state_space_size, action_space_size):
        self.Q = np.zeros((state_space_size, action_space_size))
        # self.Q = np.random.rand(state_space_size, action_space_size) * 0.01
        self.gamma = 0.9
        self.lr = 0.9



def q_learning_update(q_learning, state, action, reward, next_state, gamma, alpha):
    q_learning.Q[state, action] += alpha * (reward + (gamma * np.max(q_learning.Q[next_state, :])) - q_learning.Q[state, action])
    return q_learning


def simulate(q_learning, s, a, r, sp, num_episodes):
    print("Running simulation")

    for episode in range(num_episodes):
        for i in range(len(s)):
            state = s[i]
            action = a[i]
            reward = r[i]
            next_state = sp[i]
            gamma = q_learning.gamma
            alpha = q_learning.lr

            # print(i)
            # print(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
            q_learning = q_learning_update(q_learning, state, action, reward, next_state, gamma, alpha)
 
    return q_learning


def get_policy(q_learning):
    policy = np.argmax(q_learning.Q, axis=1)
    return policy


def state_to_index(s, grid_shape):
    s_str = str(s)
    if s < 10:  # Handle single-digit states
        x = s
        y = 0
    else:
        x = int(s_str[1])  # Correct for double-digit states
        y = int(s_str[0])
    return x, y


#count steps to goal based on a policy
def steps_to_goal(policy, start_state, goal_state, grid_size):
   
    current_state = start_state
    steps = 0
    max_steps = grid_size * grid_size * 2  # Set a maximum to prevent infinite loops if the policy is bad
    
    while current_state != goal_state and steps < max_steps:
        x, y = state_to_index(current_state, (grid_size, grid_size))
        action = policy[current_state]

        if action == 0:  # Up
            y += 1
        elif action == 1:  # Right
            x += 1
        elif action == 2:  # Down
            y -= 1
        elif action == 3:  # Left
            x -= 1
        
        # Ensure the new coordinates are within the grid bounds.
        x = np.clip(x, 0, grid_size - 1)
        y = np.clip(y, 0, grid_size - 1)

        current_state = index_to_state(x,y,(grid_size, grid_size)) # corrected
        steps += 1

    return steps if current_state == goal_state else -1  # Return -1 if goal not reached



# def visualize_optimal_actions(states, actions_opt, grid, start_state, goal_state):
def visualize_optimal_actions(path, start_state, goal_state, grid_size, grid):    
    # grid_size = 10
    grid_display = np.zeros((grid_size, grid_size), dtype=str)
    grid_display[:] = ' '
    states = [x[0] for x in path]
    actions_opt = [x[1] for x in path]

    goal_x, goal_y = state_to_index(goal_state, (grid_size, grid_size))
    start_x, start_y = state_to_index(start_state, (grid_size, grid_size))

    up = 0
    right = 1
    down = 2
    left = 3

    # Prepare a matrix for background colors
    background_colors = np.full((grid_size, grid_size), 'white', dtype=object)

    # Mark the goal position with a green background and a black 'G'
    grid_display[grid_size - 1 - goal_y, goal_x] = 'G'
    background_colors[grid_size - 1 - goal_y, goal_x] = 'green'

    # Mark the start position with a blue background
    background_colors[grid_size - 1 - start_y, start_x] = 'blue'

    # Mark obstacles with red background
    for _, row in grid.iterrows():
        if row['reward'] < 0:  # Define obstacles as cells with negative rewards
            x, y = int(row['x']), int(row['y'])
            background_colors[grid_size - 1 - y, x] = 'red'

    # Add arrows for optimal actions, including the start state
    for idx, (state, action) in enumerate(zip(states, actions_opt)):
        x, y = state_to_index(state, (grid_size, grid_size))
        arrow = None
        if action == up:
            arrow = '↑'
        elif action == down:
            arrow = '↓'
        elif action == left:
            arrow = '←'
        elif action == right:
            arrow = '→'

        # Start state arrow is black, background is blue
        if idx == 0:  # Start state
            grid_display[grid_size - 1 - y, x] = arrow
        elif background_colors[grid_size - 1 - y, x] == 'white':  # Avoid obstacles, start, and goal
            grid_display[grid_size - 1 - y, x] = arrow

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(np.zeros((grid_size, grid_size)), cmap='Greys', alpha=0)

    # Add background colors and arrows
    for i in range(grid_size):
        for j in range(grid_size):
            cell = grid_display[i, j]
            bg_color = background_colors[i, j]
            if bg_color != 'white':
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=bg_color, zorder=0))
            color = 'black'  # Default text color
            if cell == 'G':
                color = 'black'  # Goal text in black
            ax.text(j, i, cell, va='center', ha='center', color=color, fontsize=12)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(np.arange(grid_size))
    ax.set_yticklabels(np.arange(grid_size)[::-1])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.grid(False)
    plt.show()


def optimal_path(policy, start_state, goal_state, grid_size):
    path = []
    current_state = start_state
    while current_state != goal_state:
        action = policy[current_state]
        x, y = state_to_index(current_state, (grid_size, grid_size))
        if action == 0:
            y += 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            x -= 1
        next_state = index_to_state(x, y, (grid_size, grid_size))
        path.append((current_state, action, next_state))
        current_state = next_state
    return path


def param_sweep(output_file_path):
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    grid_worlds = ['map_environment/map_files/grid_world_1.csv', 'map_environment/map_files/grid_world_2.csv', 'map_environment/map_files/grid_world_3.csv']
    num_simulations = 2 # number of simulations per parameter 

    tol = 1e-3

    start_state = 0
    goal_state = 99
    max_iters = 200 # should reach the goal within max_iters steps
    
    with open(output_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['grid', 'alpha', 'gamma', 'avg steps to convergence', 'steps to convergence list'])

    count = 0 
    for alpha in alpha_values:
        for gamma in gamma_values:
            start = time.time()
            for grid_world in grid_worlds:
                grid = pd.read_csv(grid_world)
                update_trajectories(grid_world)

                dataset = 'trajectory.csv'
                file_path = f'map_environment/map_files/{dataset}'
                data = pd.read_csv(file_path)

                s = data['s'] # subtract 1 to make 0-indexed
                a = data['a'] # subtract 1 to make 0-indexed
                r = data['r']
                sp = data['sp'] # subtract 1 to make 0-indexed

                num_actions = 4
                num_states = 100
                q_learning = QLearning(num_states, num_actions)

                print(f'running simulations for parameter combo {count+1} / {len(gamma_values)*len(alpha_values)}, grid world {grid_world}')
                steps_to_convg_list = []

                for i in range(num_simulations): 

                    # rollout a single simulation
                    steps_to_convg = simulate_sweep(q_learning, s, a, r, sp, max_iters, gamma, alpha, tol)
                    steps_to_convg_list.append(steps_to_convg)
                    
                    q_learning.Q = np.zeros((num_states, num_actions))
                    
                avg_steps_to_convg = np.mean(steps_to_convg_list)

                with open(output_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([grid_world, alpha, gamma, avg_steps_to_convg, steps_to_convg_list])
            end = time.time()
            length = end-start
            print(f"one parameter combo took {length} seconds")
            count += 1
        


def simulate_sweep(q_learning, s, a, r, sp, num_episodes, gamma, alpha, tol):
    last_Q = q_learning.Q.copy()
    tol = 1e-3
    for episode in range(num_episodes):
        for i in range(len(s)):
            state = int(s[i])
            action = int(a[i])
            reward = int(r[i])
            next_state = int(sp[i])
            
            # print(i)
            # print(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
            q_learning = q_learning_update(q_learning, state, action, reward, next_state, gamma, alpha)
        
        #Check for convergence
        Q_diff = np.sum(np.abs(q_learning.Q - last_Q))
        last_Q = q_learning.Q.copy()

        if Q_diff < tol:
            # print(f"Converged after {episode + 1} episodes")
            # return q_learning
            return episode+1
        




###SINGLE SOLUTION 
grid_size = 10
start_state = 0
goal_state = 99
num_episodes = 2

grid_world = 'map_environment/map_files/grid_world_3.csv'
grid = pd.read_csv(grid_world)
update_trajectories(grid_world)

t1 = time.time()
dataset = 'trajectory.csv'
file_path = f'map_environment/map_files/{dataset}'
data = pd.read_csv(file_path)

s = data['s'] # subtract 1 to make 0-indexed
a = data['a'] # subtract 1 to make 0-indexed
r = data['r']
sp = data['sp'] # subtract 1 to make 0-indexed

num_actions = 4
num_states = 100
q_learning = QLearning(num_states, num_actions)

q_opt = simulate(q_learning, s, a, r, sp, num_episodes)       
policy = get_policy(q_opt) # add 1 to make 1-indexed
save_policy(policy, f"{dataset}.policy")

path = optimal_path(policy, start_state, goal_state, grid_size) 
step_count = len(path)
t2 = time.time()
# avg = sum(times)/len(times)

# print(f"Average runtime: {avg}")
print(f"Steps to goal: {step_count}")
# print(path)

# visualize_optimal_actions(states, actions_opt, grid, start_state, goal_state)
visualize_optimal_actions(path, start_state, goal_state, grid_size, grid)



# #PARAMETER SWEEPING
# output_file_path = 'map_environment/Q_sweep_results_3.csv'
# param_sweep(output_file_path)