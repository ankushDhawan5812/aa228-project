import matplotlib.pyplot as plt
from sample_trajectories import load_grid_with_positions, state_to_index
from map_gen import plot_grid

def load_trajectory(file_path):
    """Load trajectory from a CSV file and return as a list of tuples."""
    import csv
    trajectory = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            s, a, r, sp = map(int, row)  # Convert all values to int
            trajectory.append((s, a, r, sp))
    return trajectory

def plot_trajectory_on_grid(grid, start_pos, goal_pos, trajectory):
    """
    Plot the trajectory on the grid with start and end points labeled.

    Args:
        grid (np.ndarray): The grid to plot.
        start_pos (tuple): Coordinates of the starting position.
        goal_pos (tuple): Coordinates of the goal position.
        trajectory (list of tuples): The trajectory data as (s, a, r, sp).
    """
    # Use plot_grid to visualize the base grid with start and goal positions
    plot_grid(grid, start_pos=start_pos, goal_pos=goal_pos)

    # Convert the trajectory states into (x, y) coordinates
    grid_shape = grid.shape
    path_x = []
    path_y = []
    for s, a, r, sp in trajectory:
        x, y = state_to_index(s, grid_shape)
        path_x.append(y + 0.5)
        path_y.append(x + 0.5)

    # Plot the trajectory as a line
    plt.plot(path_x, path_y, color='blue', linestyle='-', linewidth=2, label='Trajectory')

    # Mark start and end points of the trajectory
    plt.scatter(path_x[0], path_y[0], color='green', label='Trajectory Start', zorder=5)
    plt.scatter(path_x[-1], path_y[-1], color='red', label='Trajectory End', zorder=5)

    # Add legend and title
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.title("Agent Trajectory")


    plt.show()

if __name__ == "__main__":
    # File paths
    grid_file = 'map_environment/map_files/gridworldml.csv'
    trajectory_file = 'map_environment/map_files/trajectory.csv'

    # Load the grid and positions
    grid, start_pos, goal_pos = load_grid_with_positions(grid_file)

    # Load the trajectory
    trajectory = load_trajectory(trajectory_file)

    # Plot the trajectory on the grid
    plot_trajectory_on_grid(grid, start_pos, goal_pos, trajectory)
