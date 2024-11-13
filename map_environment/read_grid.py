import numpy as np
import pandas as pd

def read_grid(file_path):
    data = pd.read_csv(file_path)
    x = data['x'].values
    y = data['y'].values
    reward = data['reward'].values
    return x, y, reward

def main():
    file_path = '/Users/ankushdhawan/Documents/Stanford/Coterm/CS238/aa228_project/map_environment/map_files/grid_world.csv'
    x, y, reward = read_grid(file_path)
    state = list(zip(x, y))
    print(state)
    print(reward)

if __name__ == "__main__":
    main()