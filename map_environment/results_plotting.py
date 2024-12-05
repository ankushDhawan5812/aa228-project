import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the CSV data
output_file_path = 'map_environment/sweep_results_2.csv'
data = pd.read_csv(output_file_path)

# Aggregate total average steps to goal across all maps for each (revisit_penalty, bias_factor) combination
agg_data = data.groupby(['revisit_penalty', 'bias_factor']).agg(
    {'avg steps to goal': 'mean', 'reached goal %': 'mean'}
).reset_index()

# Prepare data for plotting
X = agg_data['revisit_penalty'].unique()
Y = agg_data['bias_factor'].unique()
X, Y = np.meshgrid(X, Y)
Z = agg_data.pivot(index='bias_factor', columns='revisit_penalty', values='avg steps to goal').values

# Prepare data for coloring
colors = np.where(
    agg_data.pivot(index='bias_factor', columns='revisit_penalty', values='reached goal %').values == 1.0,
    'green',  # Color for reached goal % == 1.0
    'red'     # Color for other values
)

# Create the surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Loop through each point on the surface and plot a colored segment
for i in range(Z.shape[0] - 1):
    for j in range(Z.shape[1] - 1):
        verts = [
            [X[i, j], Y[i, j], Z[i, j]],
            [X[i+1, j], Y[i+1, j], Z[i+1, j]],
            [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]],
            [X[i, j+1], Y[i, j+1], Z[i, j+1]],
        ]
        ax.add_collection3d(Poly3DCollection(
            [verts],
            color=colors[i, j],
            edgecolor='k'
        ))

# Add labels, invert Y-axis, and set the title
ax.set_xlabel('Revisit Penalty')
ax.set_ylabel('Bias Factor')
ax.set_zlabel('Total Avg Steps to Goal')
ax.set_title('Surface Plot of Total Avg Steps to Goal with Conditional Coloring')
ax.invert_yaxis()

# Show the plot
plt.show()

