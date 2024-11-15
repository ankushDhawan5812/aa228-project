import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the policy from the file
policy_file = 'trajectory.policy'
policy = np.loadtxt(policy_file, dtype=int)

# Load the rewards from the file using pandas
rewards_file = 'map_environment/map_files/gridworldml.csv'
rewards_df = pd.read_csv(rewards_file)

# Define the grid size
grid_size = 10

# Create a grid to visualize the rewards
reward_grid = np.zeros((grid_size, grid_size))

# Fill the reward grid with the rewards from the dataframe
for index, row in rewards_df.iterrows():
    if row['x'] != 'start' or row['x'] != 'goal':  # Skip the header row
        x, y, reward = int(row['x']), int(row['y']), row['reward']
        print(reward)
        reward_grid[9-y, x] = reward

# Create a grid to visualize the policy
grid = np.zeros((grid_size, grid_size), dtype=str)

# Map actions to directions
action_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}

# Fill the grid with the policy directions
for state, action in enumerate(policy):
    row = state // grid_size
    col = state % grid_size
    grid[9-row, col] = action_map[action]

# Create a colormap for the rewards
cmap = ListedColormap(['white', 'green', 'red'])

# Normalize the reward values to the range [0, 2]
norm_rewards = np.zeros_like(reward_grid)
norm_rewards[reward_grid > 0] = 1
norm_rewards[reward_grid < 0] = 2

# Plot the grid
fig, ax = plt.subplots()
ax.imshow(norm_rewards, cmap=cmap, origin='upper')

# Display the policy directions
for i in range(grid_size):
    for j in range(grid_size):
        ax.text(j, i, grid[i, j], ha='center', va='center', fontsize=12, color='black')

# Hide major ticks
ax.set_xticks([])
ax.set_yticks([])

# Save the figure
plt.savefig('policy_visualization.png')

plt.show()