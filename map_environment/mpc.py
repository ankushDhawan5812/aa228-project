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

def get_reward(x, y, grid):
    reward_value = grid[(grid['x'] == x) & (grid['y'] == y)]['reward'].values[0]
    return reward_value

def get_reward_from_neighbors(neighborhood, grid, previously_visited, revisit_penalty):
    rewards = {}
    n = neighborhood.keys()
    grid_shape = (10, 10)
    for neighbor in n:
        neighbor = int(neighbor)
        x, y = state_to_index(neighbor, grid_shape)
        rewards[neighbor] = get_reward(x, y, grid)
        if neighbor in previously_visited: # Penalize revisiting previously visited states with large negative reward
            num_instances = previously_visited.count(neighbor)
            # num_instances = 1
            rewards[neighbor] = revisit_penalty * num_instances # compound penalty based on number of times revisited
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

def mpc_loop(start_state, max_steps, grid, goal_state=99, revisit_penalty=-100, shuffle=False):
    states = []
    state = start_state
    states.append(start_state) # appending start state to already visited
    actions_opt = []
    for i in range(max_steps):
        n = get_neighborhood(state, shuffle)
        actions = list(n.values())
        neighbor_states = list(n.keys())
        rewards = list(get_reward_from_neighbors(n, grid, states, revisit_penalty).values())
        plan_horizon = get_plan(state, neighbor_states, actions, rewards)
        
        state = execute_action(state, plan_horizon[0]) # execute action with first action of the horizon 
        print("next state: ", state)
        print("actions: ", plan_horizon[0])
        if state in states:
            print(f"Revisiting state: {state}...")

        states.append(state)
        actions_opt.append(plan_horizon[0])
        
        if state == goal_state:
            print(f"Reached goal in {i} steps")
            return states, actions_opt
        
    return states, actions_opt

def visualize_optimal_actions(states, actions_opt):
    grid_size = 10
    grid = np.zeros((grid_size, grid_size), dtype=str)
    grid[:] = ' '
    goal_state = 99
    goal_x, goal_y = state_to_index(goal_state, (grid_size, grid_size))
    grid[grid_size - 1 - goal_y, goal_x] = 'G'

    for state, action in zip(states, actions_opt):
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
        # print(f"arrow: {arrow}")
        grid[grid_size - 1 - y, x] = arrow # Adjust for coordinate system

    fig, ax = plt.subplots()
    ax.matshow(np.zeros((grid_size, grid_size)), cmap='Greys')

    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, grid[i, j], va='center', ha='center')

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(np.arange(grid_size))
    ax.set_yticklabels(np.arange(grid_size)[::-1])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.show()

def get_policy(q_learning):
    policy = np.argmax(q_learning.Q, axis=1)
    return policy

grid = pd.read_csv('map_environment/map_files/gridworldad.csv')
start_state = 0
goal_state = 99
max_iters = 200
revisit_penalty = -1
shuffle = True
states, actions_opt = mpc_loop(start_state, max_iters, grid, goal_state, revisit_penalty, shuffle)

print("STATES: ", states)
print("ACTIONS: ", actions_opt)
visualize_optimal_actions(states, actions_opt)