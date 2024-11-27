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
grid_shape = (10, 10)

def save_policy(policy, file_path):
    print(type(policy))
    with open(file_path, "w") as f:
        for si in range(100):
            f.write(f"{policy[si]}\n")

class SarsaLambda:
    def __init__(self, state_space_size, action_space_size):
        self.S = range(state_space_size)
        self.A = range(action_space_size)
        # self.Q = np.random.rand(state_space_size, action_space_size) * 0.01     # Randomly initialized value function
        # self.Q = np.zeros((state_space_size, action_space_size))              # Zeroized initialized value function
        self.Q = np.full((state_space_size, action_space_size), -100.0) 
        self.gamma = 0.95                                                       #reward discount rate (γ)
        self.lr = 0.09                                                          #learning rate (α)
        self.lam = 0.5                                                         #trace decay rate (λ)
        self.N = np.zeros((state_space_size, action_space_size))                #trace count
        self.last = None                                                        #last experience

sarsaLam = SarsaLambda(num_states, num_actions)



# def epsilon_greedy_policy(Q, state, epsilon=0.1):
#     """
#     Select an action using epsilon-greedy policy.
#     """
#     if np.random.rand() < epsilon:
#         return np.random.randint(len(Q[state]))  # Explore: random action
#     else:
#         return np.argmax(Q[state])  # Exploit: best action



# Sarsa Lambda that does not include the next_action input
# def sarsaLam_update(sarsaLam, state, action, reward, next_state):
#     """
#     Sarsa Lamda update rule.
#     """
#     gamma = sarsaLam.gamma
#     lam = sarsaLam.lam
#     alpha = sarsaLam.lr
#     if sarsaLam.last != None: 
#         s_last, a_last, r_last = sarsaLam.last
#         sarsaLam.N[s_last, a_last] += 1      #increment visit count for last state/action pair
#         delta = r_last + gamma * sarsaLam.Q[state, action] - sarsaLam.Q[s_last, a_last]
#         for s in sarsaLam.S:
#             for a in sarsaLam.A:
#                 s_x, s_y = state_to_index(s, (10, 10))  # Assuming grid is 10x10
#                 valid_action = True
#                 if a == 0 and s_y == 9:  # North
#                     valid_action = False
#                 elif a == 1 and s_x == 9: # East
#                     valid_action = False
#                 elif a == 2 and s_y == 0: # South
#                     valid_action = False
#                 elif a == 3 and s_x == 0: # West
#                     valid_action = False
                
#                 if valid_action:
#                     sarsaLam.Q[state, action] += alpha*delta*sarsaLam.N[state, action]
#                     sarsaLam.N[state, action] *= gamma*lam
#     else:    
#         sarsaLam.N.fill(0.0)
    
#     sarsaLam.last = (state, action, reward)
#     return sarsaLam


def sarsaLam_update(sarsaLam, state, action, reward, next_state):
    gamma = sarsaLam.gamma
    lam = sarsaLam.lam
    alpha = sarsaLam.lr

    if sarsaLam.last is not None:
        s_last, a_last, r_last = sarsaLam.last
        sarsaLam.N[s_last, a_last] += 1
        delta = r_last + gamma * sarsaLam.Q[state, action] - sarsaLam.Q[s_last, a_last]

        for s in sarsaLam.S:
            valid_actions = check_border(s)
            for a in valid_actions:
                sarsaLam.Q[s, a] += alpha * delta * sarsaLam.N[s, a]  # Update Q for valid s, a
                sarsaLam.N[s, a] *= gamma * lam  # Decay trace for valid s, a
        
    else:
        sarsaLam.N.fill(0.0)

    sarsaLam.last = (state, action, reward)
    return sarsaLam


def check_border(s):
    x, y = state_to_index(s, grid_shape)
    valid_actions = []
    if y < grid_shape[0] - 1:  # North
        valid_actions.append(0)
    if x < grid_shape[1] - 1:  # East
        valid_actions.append(1)
    if y > 0:  # South
        valid_actions.append(2)
    if x > 0:  # West
        valid_actions.append(3)
    return valid_actions

def state_to_index(s, grid_shape):
    s_str = str(s)
    if s < 10:  # Handle single-digit states
        x = s
        y = 0
    else:
        x = int(s_str[1])  # Correct for double-digit states
        y = int(s_str[0])
    return x, y


#Sarsa Lambda with the next action input
# def sarsaLam_update(sarsaLam, state, action, reward, next_state, next_action):
#     """
#     Sarsa Lamda update rule.
#     """
#     gamma = sarsaLam.gamma
#     lam = sarsaLam.lam
#     alpha = sarsaLam.lr
#     if sarsaLam.last != None: 
#         s_last, a_last, r_last = sarsaLam.last
#         sarsaLam.N[s_last, a_last] += 1      #increment visit count for last state/action pair
#         delta = r_last + gamma * sarsaLam.Q[next_state, next_action] - sarsaLam.Q[s_last, a_last]
#         for s in sarsaLam.S:
#             for a in sarsaLam.A:
#                 sarsaLam.Q[next_state, next_action] += alpha*delta*sarsaLam.N[next_state, next_action]
#                 sarsaLam.N[next_state, next_action] *= gamma*lam

#     else:    
#         sarsaLam.N.fill(0.0)
    
#     sarsaLam.last = (state, action, reward)
#     return sarsaLam



def simulate(sarsaLam, s, a, r, sp, num_episodes=10, epsilon=0.1):
    """
    Simulate episodes for Sarsa Lambda learning.
    """
    print("Running simulation")
    for episode in range(num_episodes):
        for i in range(len(s)):
            state = s[i]
            action = a[i]
            reward = r[i]
            next_state = sp[i]
            
            # Choose next action using epsilon-greedy policy
            # next_action = epsilon_greedy_policy(sarsaLam.Q, next_state, epsilon)
            
            # Perform Sarsa Lambda update
            # sarsaLam = sarsaLam_update(sarsaLam, state, action, reward, next_state, next_action)
            sarsaLam = sarsaLam_update(sarsaLam, state, action, reward, next_state) #not using the epsilon greedy policy 

    return sarsaLam

def get_policy(sarsaLam):
    """
    Extract the optimal policy from the learned Q-table.
    """
    policy = np.argmax(sarsaLam.Q, axis=1)
    return policy

# Simulate Sarsa Lambda
sarsaLam_opt = simulate(sarsaLam, s, a, r, sp)
policy = get_policy(sarsaLam_opt)

# Save the resulting policy
save_policy(policy, f"{dataset}_sarsaLam.policy")

t2 = time.time()
print(f"Runtime: {t2-t1}")
