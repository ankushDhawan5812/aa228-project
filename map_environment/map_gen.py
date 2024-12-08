import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv

def create_grid(size=(10, 10), start_pos=(0, 0), goal_pos=(9, 9), collision_positions=None):
    grid = np.zeros(size)

    # Assign rewards
    grid[start_pos] = 0  # Starting position (neutral)
    grid[goal_pos] = 10  # Large positive reward at the goal
    if collision_positions:
        for pos in collision_positions:
            grid[pos] = -10  # Large negative reward at collision points

    return grid

def plot_grid(grid, start_pos, goal_pos):
    grid_shape = grid.shape
    cmap = ListedColormap(['lightcoral', 'white', 'lightgreen'])
    norm = plt.Normalize(vmin=-10, vmax=10)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, norm=norm, origin="lower", extent=[0, grid_shape[1], 0, grid_shape[0]])
    plt.colorbar(label='Rewards', ticks=[-10, 0, 10])

    # Annotate the start and goal
    plt.text(start_pos[1] + 0.5, start_pos[0] + 0.5, 'S', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')
    plt.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')

    # Set grid lines without changing the tick positions
    plt.xticks(np.arange(0, grid_shape[1], 1), labels=[""] * grid_shape[1])
    plt.yticks(np.arange(0, grid_shape[0], 1), labels=[""] * grid_shape[0])
    plt.grid(which="major", color="gray", linestyle="--", linewidth=0.5)

    # Add shifted axis labels
    for i in range(grid_shape[1]):
        plt.text(i + 0.5, -0.5, str(i), ha='center', va='center', fontsize=10, transform=plt.gca().transData)
    for i in range(grid_shape[0]):
        plt.text(-0.5, i + 0.5, str(i), ha='center', va='center', fontsize=10, transform=plt.gca().transData)

    plt.title("Grid World")

def save_grid(grid, start_pos, goal_pos, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'reward'])
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                writer.writerow([x, y, grid[x, y]])
        writer.writerow(['start', start_pos[0], start_pos[1]])
        writer.writerow(['goal', goal_pos[0], goal_pos[1]])

if __name__ == "__main__":
    # Parameters
    grid_size = (10, 10)
    start_position = (0, 0)
    goal_position = (9, 9)
    collisions = [(3, 3), (3, 4), (4, 3), (4, 4)]

    # Create the grid
    grid = create_grid(size=grid_size, start_pos=start_position, goal_pos=goal_position, collision_positions=collisions)

    # Plot the grid
    plot_grid(grid, start_pos=start_position, goal_pos=goal_position)
    plt.show()

    # Save the grid to a CSV file
    save_grid(grid, start_position, goal_position, 'map_environment/map_files/gridworldml.csv')
