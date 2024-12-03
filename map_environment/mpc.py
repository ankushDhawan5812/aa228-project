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
sp = data['sp'] # subtract 1 to make 0-indexed

def save_policy(policy, file_path):
    print(type(policy))
    with open(file_path, "w") as f:
        for si in range(100):
            f.write(f"{policy[si]}\n")

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
            # print(i)
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

def get_neighborhood(state):
    grid_shape = (10, 10)
    x, y = state_to_index(state, grid_shape)
    neighborhood = {}
    if x > 0:
        neighborhood[state - 1] = 'left'
    if x < grid_shape[1] - 1:
        neighborhood[state + 1] = 'right'
    if y > 0:
        neighborhood[state - 10] = 'down'
    if y < grid_shape[0] - 1:
        neighborhood[state + 10] = 'up'
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

def get_reward_from_neighbors(neighborhood, grid, previously_visited):
    rewards = {}
    n = neighborhood.keys()
    grid_shape = (10, 10)
    for neighbor in n:
        neighbor = int(neighbor)
        x, y = state_to_index(neighbor, grid_shape)
        rewards[neighbor] = get_reward(x, y, grid)
        if neighbor in previously_visited: # Penalize revisiting previously visited states with large negative reward
            rewards[neighbor] = -100
    print(f"Rewards: {rewards}")
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

def mpc_loop(start_state, max_steps, grid):
    states = []
    state = start_state
    actions_opt = []
    for i in range(max_steps):
        print(f"State: {state}")
        # print(f"Step: {i}")
        n = get_neighborhood(state)
        # print(f"Neighborhood: {n}")
        actions = list(n.values())
        # print(f"Actions: {actions}")
        neighbor_states = list(n.keys())
        # print(f"Neighbor states: {neighbor_states}")
        rewards = list(get_reward_from_neighbors(n, grid, states).values())
        # print(f"Rewards: {rewards}")
        plan_horizon = get_plan(state, neighbor_states, actions, rewards)
        state = execute_action(state, plan_horizon[0]) # execute action with first action of the horizon 
        if state in states:
            print(f"Revisiting state: {state}")
        if state == 99:
            print("Reached goal")
        states.append(state)
        actions_opt.append(plan_horizon[0])
    return states, actions_opt

def get_policy(q_learning):
    policy = np.argmax(q_learning.Q, axis=1)
    return policy

grid = pd.read_csv('map_environment/map_files/gridworldad.csv')

states, actions_opt = mpc_loop(0, 100, grid)
print(states)
print(actions_opt)

r = get_reward(0, 7, grid)
print(r)

# x = 4
# y = 3
# print(f"Reward at ({x}, {y}): {get_reward(x, y, grid)}")

# state = 10
# n = get_neighborhood(state)
# print(f"Neighborhood of {state}: {n}")

# next_state = execute_action(state, 'left')
# print(next_state)
# r_neighbors = get_reward_from_neighbors(n, grid)
# print(f"Rewards of neighbors: {r_neighbors}")

# grid_shape = (10, 10)
# start_state = 0
# max_steps = 10
# state = 55
# print(f"{state} state state: {state_to_index(state, grid_shape)}")
# print(f"{state} state neighborhood: {get_neighborhood(state)}")

# q_opt = simulate(q_learning, s, a, r, sp)
# policy = get_policy(q_opt) # add 1 to make 1-indexed


# save_policy(policy, f"{dataset}.policy")


# t2 = time.time()

# print(f"Runtime: {t2-t1}")

