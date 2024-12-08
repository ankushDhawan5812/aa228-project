import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

t1 = time.time()

dataset = 'trajectory.csv'
file_path = f'map_environment/map_files/{dataset}'
data = pd.read_csv(file_path)

s = data['s'] # subtract 1 to make 0-indexed
a = data['a'] # subtract 1 to make 0-indexed
r = data['r']
sp = data['sp'] # subtract 1 to make 0-indexeds

class QLearning:
    def __init__(self, state_space_size, action_space_size):
        self.Q = np.zeros((state_space_size, action_space_size))
        # self.Q = np.random.rand(state_space_size, action_space_size) * 0.01

        self.gamma = 0.95
        self.lr = 0.09

def q_learning_update(q_learning, state, action, reward, next_state):
    q_learning.Q[state, action] += q_learning.lr * (reward + (q_learning.gamma * np.max(q_learning.Q[next_state, :])) - q_learning.Q[state, action])
    return q_learning

def simulate(q_learning, s, a, r, sp, num_episodes=10):
    # print("Running simulation")
    for episode in range(num_episodes):
        for i in range(len(sp)):
            action = a[i]
            reward = r[i]
            next_state = sp[i]
            # print(i)
            # print(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
            q_learning = q_learning_update(q_learning, s, action, reward, next_state)

    return q_learning

def state_to_index(s, grid_shape):
    s_str = str(s)
    if s < 10:  # Handle single-digit states
        x = s
        y = 0
    else:
        x = int(s_str[1])  # Correct for double-digit states
        y = int(s_str[0])
    return x, y

def get_neighborhood(state, shuffle):
    grid_shape = (10, 10)
    x, y = state_to_index(state, grid_shape)
    neighborhood = {}
    if y < grid_shape[0] - 1:
        neighborhood[state + 10] = 'up'
    if y > 0:
        neighborhood[state - 10] = 'down'
    if x < grid_shape[1] - 1:
        neighborhood[state + 1] = 'right'
    if x > 0:
        neighborhood[state - 1] = 'left'
    if shuffle:
        neighborhood_items = list(neighborhood.items())
        np.random.shuffle(neighborhood_items)
        neighborhood = dict(neighborhood_items)
    
    return neighborhood

def execute_action(state, action):
    if action == 'left':
        assert state > 0, "Invalid action: cannot move left from the leftmost state"
        return state - 1
    if action == 'right':
        assert state < 99, "Invalid action: cannot move right from the rightmost state"
        return state + 1
    if action == 'down':
        assert state > 9, "Invalid action: cannot move up from the bottommost state"
        return state - 10
    if action == 'up':
        assert state < 90, "Invalid action: cannot move up from the topmost state"
        return state + 10
    assert False, f"Invalid action: {action}"

def get_rewards(next_states):
    rewards = []
    grid_shape = (10, 10)
    for next_state in next_states:
        x, y = state_to_index(next_state, grid_shape)
        rewards.append(grid[x, y])
    return rewards

def get_reward(x, y, grid, obstacle_weight):
    reward_value = grid[(grid['x'] == x) & (grid['y'] == y)]['reward'].values[0]
    if reward_value < 0:
        reward_value *= obstacle_weight
    return reward_value

def get_reward_from_neighbors(neighborhood, grid, previously_visited, revisit_penalty, goal_state, bias_factor, obstacle_weight):
    rewards = {}
    n = neighborhood.keys()
    grid_shape = (10, 10)
    goal_x, goal_y = state_to_index(goal_state, grid_shape)

    for neighbor in n:
        neighbor = int(neighbor)
        x, y = state_to_index(neighbor, grid_shape)
        rewards[neighbor] = get_reward(x, y, grid, obstacle_weight) # base reward
        if neighbor in previously_visited: # Penalize revisiting previously visited states with large negative reward
            num_instances = previously_visited.count(neighbor)
            # num_instances = 1
            rewards[neighbor] -= revisit_penalty * num_instances # compound penalty based on number of times revisited

        # Add reward shaping (bias term) based on distance to goal
        neighbor_x, neighbor_y = state_to_index(neighbor, grid_shape)
        distance_to_goal = abs(goal_x - neighbor_x) + abs(goal_y - neighbor_y) #L1 distance
        rewards[neighbor] -= bias_factor * distance_to_goal # negative reward for larger distance to goal
    # print(f"Rewards: {rewards}")
    return rewards

def get_plan(cur_state, neighbor_states, actions, rewards):
    a = list(range(len(actions)))
    r = rewards
    sp = list(range(len(neighbor_states)))

    q_learning = QLearning(len(s), len(a))
    q_learning = simulate(q_learning, cur_state, a, r, sp, num_episodes=100)
    policy = get_policy(q_learning)
    best_action_index = policy[cur_state]
    best_action = actions[best_action_index]
    return [best_action]

def mpc_loop(start_state, max_steps, grid, goal_state=99, revisit_penalty=1, bias_factor=1, obstacle_weight=1, shuffle=False):
    states = []
    state = start_state
    states.append(start_state) # appending start state to already visited
    actions_opt = []
    reached_goal = False
    hit_obstacle = False

    for i in range(max_steps):
        n = get_neighborhood(state, shuffle)
        actions = list(n.values())
        neighbor_states = list(n.keys())
        rewards = list(get_reward_from_neighbors(n, grid, states, revisit_penalty, goal_state, bias_factor, obstacle_weight).values())
        plan_horizon = get_plan(state, neighbor_states, actions, rewards)
        
        state = execute_action(state, plan_horizon[0]) # execute action with first action of the horizon 
        # print("next state: ", state)
        # print("actions: ", plan_horizon[0])


        if state in states:
            # print(f"Revisiting state: {state}...")
            pass

        # Check if the current state is an obstacle
        obstacle = grid[(grid['x'] == state % 10) & (grid['y'] == state // 10)]['reward'].values[0] < 0
        if obstacle:
            # print(f"Hit an obstacle at state {state}!")
            hit_obstacle = True
            break

        states.append(state)
        actions_opt.append(plan_horizon[0])
        
        if state == goal_state:
            # print(f"Reached goal in {i} steps")
            reached_goal = True
            return states, actions_opt, reached_goal, hit_obstacle, i
        
    return states, actions_opt, reached_goal, hit_obstacle, 0 


def visualize_optimal_actions(states, actions_opt, grid, start_state, goal_state):
    grid_size = 10
    grid_display = np.zeros((grid_size, grid_size), dtype=str)
    grid_display[:] = ' '

    goal_x, goal_y = state_to_index(goal_state, (grid_size, grid_size))
    start_x, start_y = state_to_index(start_state, (grid_size, grid_size))

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
        if action == 'up':
            arrow = '↑'
        elif action == 'down':
            arrow = '↓'
        elif action == 'left':
            arrow = '←'
        elif action == 'right':
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

def get_policy(q_learning):
    policy = np.argmax(q_learning.Q, axis=1)
    return policy

# grid = pd.read_csv('map_environment/map_files/grid_world_3.csv')
# start_state = 0
# goal_state = 99
# max_iters = 1000
# revisit_penalty = 1
# bias_factor = 1
# obstacle_weight = 10
# shuffle = True
# states, actions_opt, reached_goal, hit_obstacle, steps_to_goal = mpc_loop(start_state, max_iters, grid, goal_state, revisit_penalty, bias_factor, obstacle_weight, shuffle)

# print("STATES: ", states)
# print("ACTIONS: ", actions_opt)
# print(f"Reached goal? {reached_goal}")
# print(f'Hit obstacle? {hit_obstacle}')


# visualize_optimal_actions(states, actions_opt, grid, start_state, goal_state)
