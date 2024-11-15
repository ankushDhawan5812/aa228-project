import numpy as np
import csv
import random

def load_grid_with_positions(file_path):
    """Load grid, start, and goal positions from a CSV file."""
    grid = []
    start_pos = None
    goal_pos = None
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'x':  # Skip header row
                continue
            elif row[0] == 'start':
                start_pos = (int(row[1]), int(row[2]))
            elif row[0] == 'goal':
                goal_pos = (int(row[1]), int(row[2]))
            else:
                x, y, reward = map(float, row)  # Convert values to float
                x, y, reward = int(x), int(y), int(reward)
                while len(grid) <= x:
                    grid.append([0] * (y + 1))
                while len(grid[x]) <= y:
                    grid[x].append(0)
                grid[x][y] = reward
    return np.array(grid), start_pos, goal_pos

def index_to_state(x, y, grid_shape):
    idx_str = f"{y}{x}"
    idx = int(idx_str)
    print(f"x: {x}, y: {y}, index: {idx} ")
    return idx

def state_to_index(s, grid_shape):
    s_str = str(s)
    x = int(s_str[1])
    y = int(s_str[0])
    return x, y

# def index_to_state(x, y, grid_shape):
#     """Convert (x, y) coordinates to a single state index with 1-based indexing."""
#     print(f"x: {x}, y: {y}, index: {np.ravel_multi_index((x, y), grid_shape)} ")

#     return np.ravel_multi_index((x, y), grid_shape)

# def state_to_index(s, grid_shape):
#     """Convert a single state index back to (x, y) coordinates with 1-based indexing."""
#     return np.unravel_index(s, grid_shape)

def generate_random_trajectory(grid, start_pos=None, num_steps=10):
    """Generate a random trajectory of the agent through the environment."""
    grid_shape = grid.shape
    trajectory = []

    # Start from the specified start position or a random one
    if start_pos:
        x, y = start_pos
    else:
        x, y = random.randint(0, grid_shape[0] - 1), random.randint(0, grid_shape[1] - 1)
    s = index_to_state(x, y, grid_shape)

    for _ in range(num_steps):
        # Choose a random action
        valid_actions = []
        if y < grid_shape[0] - 1:  # North
            valid_actions.append(0)
        if x < grid_shape[1] - 1:  # East
            valid_actions.append(1)
        if y > 0:  # South
            valid_actions.append(2)
        if x > 0:  # West
            valid_actions.append(3)

        action = random.choice(valid_actions)
        dx, dy = 0, 0
        if action == 0:  # North
            dy = +1
        elif action == 1:  # East
            dx = 1
        elif action == 2:  # South
            dy = -1
        elif action == 3:  # West
            dx = -1

        # Update position
        new_x, new_y = x + dx, y + dy
        sp = index_to_state(new_x, new_y, grid_shape)
        reward = grid[x, y]

        # Append to trajectory
        trajectory.append([s, action, reward, sp])

        # Move to the new state
        x, y, s = new_x, new_y, sp

    return trajectory

def save_trajectory(trajectory, file_path):
    """Save the trajectory to a CSV file."""
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s', 'a', 'r', 'sp'])
        writer.writerows(trajectory)

if __name__ == "__main__":
    # File path for the grid
    grid_file = 'map_environment/map_files/gridworldml.csv'
    trajectory_file = 'map_environment/map_files/trajectory.csv'

    # Load the grid and positions
    grid, start_pos, _ = load_grid_with_positions(grid_file)

    # Generate a random trajectory
    trajectory = generate_random_trajectory(grid, start_pos=start_pos, num_steps=50000)

    # Save the trajectory to a CSV file
    save_trajectory(trajectory, trajectory_file)
