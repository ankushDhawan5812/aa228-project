import pandas as pd
import numpy as np
import time

t1 = time.time()

dataset = 'trajectory'
file_path = f'map_environment/map_files/{dataset}.csv'
data = pd.read_csv(file_path)

# s = data['s'] - 1 # Uncomment if data uses 1-indexing and needs conversion to 0-indexing
# a = data['a'] - 1
# sp = data['sp'] - 1

s = data['s']  # Comment out if adjusting to 0-indexed
a = data['a']
r = data['r']
sp = data['sp']

num_actions = 4
num_states = 100

def save_policy(policy, file_path):
    print(type(policy))
    with open(file_path, "w") as f:
        for si in range(100):
            f.write(f"{policy[si]}\n")

class Sarsa:
    def __init__(self, state_space_size, action_space_size):
        self.Q = np.random.rand(state_space_size, action_space_size) * 0.01
        self.gamma = 0.95
        self.lr = 0.09

sarsa = Sarsa(num_states, num_actions)

def epsilon_greedy_policy(Q, state, epsilon=0.1):
    """
    Select an action using epsilon-greedy policy.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q[state]))  # Explore: random action
    else:
        return np.argmax(Q[state])  # Exploit: best action
    # return np.argmax(Q[state])


def sarsa_update(sarsa, state, action, reward, next_state, next_action):
    """
    Sarsa update rule.
    """
    sarsa.Q[state, action] += sarsa.lr * (
        reward + (sarsa.gamma * sarsa.Q[next_state, next_action]) - sarsa.Q[state, action]
    )
    return sarsa

def simulate(sarsa, s, a, r, sp, num_episodes=10, epsilon=0.1):
    """
    Simulate episodes for Sarsa learning.
    """
    print("Running simulation")
    for episode in range(num_episodes):
        for i in range(len(s)):
            state = s[i]
            action = a[i]
            reward = r[i]
            next_state = sp[i]
            
            # Choose next action using epsilon-greedy policy
            next_action = epsilon_greedy_policy(sarsa.Q, next_state, epsilon)
            
            # Perform Sarsa update
            sarsa = sarsa_update(sarsa, state, action, reward, next_state, next_action)

    return sarsa

def get_policy(sarsa):
    """
    Extract the optimal policy from the learned Q-table.
    """
    policy = np.argmax(sarsa.Q, axis=1)
    return policy

# Simulate Sarsa
sarsa_opt = simulate(sarsa, s, a, r, sp)
policy = get_policy(sarsa_opt)

# Save the resulting policy
save_policy(policy, f"{dataset}_sarsa.policy")

t2 = time.time()
print(f"Runtime: {t2-t1}")
