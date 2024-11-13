import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv

# Initialize a 10x10 grid with all zeros
grid = np.zeros((10, 10))

# Define positions
start_pos = (0, 0)  # Starting position at the top-left
goal_pos = (9, 9)   # Goal position at the bottom-right
collision_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]  # Example obstacles

# Assign rewards
grid[start_pos] = 0    # Starting position (neutral)
grid[goal_pos] = 10    # Large positive reward at the goal
for pos in collision_positions:
    grid[pos] = -10    # Large negative reward at collision points

# Define color map
cmap = ListedColormap(['white', 'lightcoral', 'lightgreen'])
bounds = [-10, -1, 1, 9, 11]
norm = plt.Normalize(vmin=-10, vmax=10)

# Create the plot with corrected alignment and adjusted coordinates
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap=cmap, norm=norm, origin="lower", extent=[-1, 9, -1, 9])
plt.colorbar(label='Rewards', ticks=[-10, 0, 10])

# Annotate the start and goal with adjusted positions
plt.text(start_pos[1] - 0.5, start_pos[0] - 0.5, 'S', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')
plt.text(goal_pos[1] - 0.5, goal_pos[0] - 0.5, 'G', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')

# Set grid lines and labels for better alignment
plt.xticks(np.arange(-1, 10, 1))
plt.yticks(np.arange(-1, 10, 1))
plt.gca().set_xticks(np.arange(-1, 10, 1) + 0.5, minor=True)
plt.gca().set_yticks(np.arange(-1, 10, 1) + 0.5, minor=True)
plt.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)

# Title and display
plt.title("10x10 Grid World Starting at (-1, -1)")
plt.show()

# Save the grid world to a CSV file
csv_filename = '/Users/ankushdhawan/Documents/Stanford/Coterm/CS238/aa228_project/map_environment/map_files/grid_world.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'reward'])
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            writer.writerow([x, y, grid[x, y]])