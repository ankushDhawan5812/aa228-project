import pandas as pd
import numpy as np
import time

t1 = time.time()

dataset = 'trajectory.csv'
file_path = f'/Users/ankushdhawan/Documents/Stanford/Coterm/CS238/aa228_project/map_environment/map_files{dataset}.csv'
data = pd.read_csv(file_path)

# print(data.head())

s = data['s'] - 1 # subtract 1 to make 0-indexed
a = data['a'] - 1 # subtract 1 to make 0-indexed
r = data['r']
sp = data['sp'] - 1 # subtract 1 to make 0-indexed

num_actions = 4
num_states = 100

def save_policy(policy, file_path):
    print(type(policy))
    with open(file_path, "w") as f:
        for si in range(100):
            f.write(f"{policy[si]}\n")

class QLearning:
    def __init__(self, state_space_size, action_space_size):
        # self.Q = np.zeros((state_space_size, action_space_size))
        self.Q = np.random.rand(state_space_size, action_space_size) * 0.01

        self.gamma = 0.95
        self.lr = 0.09

q_learning = QLearning(num_states, num_actions)

def q_learning_update(q_learning, state, action, reward, next_state):
    q_learning.Q[state, action] += q_learning.lr * (reward + (q_learning.gamma * np.max(q_learning.Q[next_state, :])) - q_learning.Q[state, action])

    return q_learning

def simulate(q_learning, s, a, r, sp, num_episodes=100):
    print(len(s))
    for episode in range(num_episodes):
        for i in range(len(s)):
            state = s[i]
            action = a[i]
            reward = r[i]
            next_state = sp[i]
            # print(i)
            # print(next_state)
            q_learning = q_learning_update(q_learning, state, action, reward, next_state)

    return q_learning

def get_policy(q_learning):
    policy = np.argmax(q_learning.Q, axis=1)
    return policy

q_opt = simulate(q_learning, s, a, r, sp)
policy = get_policy(q_opt) + 1 # add 1 to make 1-indexed


save_policy(policy, f"{dataset}.policy")


t2 = time.time()

print(f"Runtime: {t2-t1}")
