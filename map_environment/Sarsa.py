import pandas as pd
import numpy as np
import time

t1 = time.time()

dataset = 'trajectory.csv'
file_path = f'map_environment/map_files/{dataset}'
data = pd.read_csv(file_path)

# print(data.head())

# s = data['s'] - 1 # subtract 1 to make 0-indexed
# a = data['a'] - 1 # subtract 1 to make 0-indexed
# r = data['r']
# sp = data['sp'] - 1 # subtract 1 to make 0-indexed

s = data['s'] # subtract 1 to make 0-indexed
a = data['a'] # subtract 1 to make 0-indexed
r = data['r']
sp = data['sp'] # subtract 1 to make 0-indexed

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
        self.last = None #most recent experience tuple (s, a, r)

sarsa = Sarsa(num_states, num_actions)


def sarsa_update(sarsa, state, action, reward, next_state):
    if sarsa.last != None:
        # gamma = sarsa.gamma
        # Q = sarsa.Q
        # lr = sarsa.lr
        # last = sarsa.last

        sarsa.Q[sarsa.last['state'], sarsa.last['action']] += sarsa.lr * ( sarsa.last['reward'] + sarsa.gamma * sarsa.Q[state, action] - sarsa.Q[sarsa.last['state'], sarsa.last['action']] )
    sarsa.last = {'state': state, 'action': action, 'reward': reward}
    
    return sarsa



def simulate(sarsa, s, a, r, sp, num_episodes=100): 
    print("Running simulation")
    for episode in range(num_episodes):
        for i in range(len(s)):
            # print(i)
            state = s[i]
            action = a[i]
            reward = r[i]
            next_state = sp[i]
            # print(i)
            # print(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
            sarsa = sarsa_update(sarsa, state, action, reward, next_state)

    return sarsa


def get_policy(sarsa):
    policy = np.argmax(sarsa.Q, axis=1)
    return policy

q_opt = simulate(sarsa, s, a, r, sp)
policy = get_policy(q_opt) # add 1 to make 1-indexed


save_policy(policy, f"{dataset}.policy")


t2 = time.time()

print(f"Runtime: {t2-t1}")
