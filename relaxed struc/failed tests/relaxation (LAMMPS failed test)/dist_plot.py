''' #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:34:35 2023

@author: alejandrosoto
"""

import numpy as np
import matplotlib.pyplot as plt

# Load oxDNA structure information from .dat file
dat_file = 'output.dat'

# Read the relevant information from the .dat file
# Replace the following lines with your specific code to extract data
with open(dat_file, 'r') as f:
    # Assuming each line in the .dat file contains relevant information
    data = [float(line.strip()) for line in f]

# Assuming data now contains the relevant distances (replace this with your actual data)
distances = np.array(data)

# Sort distances and plot
sorted_distances = np.sort(distances)
plt.plot(range(1, len(sorted_distances) + 1), sorted_distances)
plt.title('Distance Distribution')
plt.xlabel('Data Point Index')
plt.ylabel('Distance')
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
'''

"""
Created on Fri Dec 8 11:34:35 2023

@author: alejandrosoto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 8 11:34:35 2023

@author: alejandrosoto
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Load oxDNA structure information from .dat file
dat_file = 'output.dat'

# Create a list to store positions
positions = []

# Read the relevant information from the .dat file
with open(dat_file, 'r') as f:
    for line in f:
        # Split the line into values
        values = line.strip().split()
        if len(values) >= 15:  # Assuming 15 columns in total
            # Assuming the first three columns represent the position of the centre of mass (x, y, z)
            position = [float(val) for val in values[:3]]
            positions.append(position)

# Convert positions to a numpy array for easier manipulation
positions = np.array(positions)

# Define a function to calculate the distance to the k-th nearest neighbor
def calculate_kn_distance(X, k):
    kn_distance = []
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances_k, _ = nbrs.kneighbors(X)
    for i in range(len(X)):
        kn_distance.append(distances_k[i, -1])  # Distance to the k-th nearest neighbor
    return kn_distance

# Calculate distances to the k-th nearest neighbor for different values of k
max_k = 10  # You can adjust this based on your data
distances = []
for k in range(1, max_k + 1):
    eps_dist = calculate_kn_distance(positions, k)
    distances.append(np.mean(eps_dist))  # Use the mean distance for simplicity

# Find the "elbow" point in the curve
diff = np.diff(distances, 2)  # Second derivative
elbow_point = np.argmax(diff) + 2  # Add 2 to get the correct index in the original array

# Recommended epsilon value
recommended_epsilon = distances[elbow_point - 1]  # Subtract 1 to get the correct index

# Plot the distances
plt.subplot(1, 2, 1)
plt.plot(range(1, max_k + 1), distances, marker='o')
plt.scatter(elbow_point, recommended_epsilon, color='red', label='Recommended Epsilon')
plt.title('Mean Distance to k-th Nearest Neighbor')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Mean Distance')
plt.legend()

# Calculate and plot the histogram of average distances
plt.subplot(1, 2, 2)
avg_distances = calculate_kn_distance(positions, 4)  # Distances to the 4th nearest neighbor
plt.hist(avg_distances, bins=20)
plt.title('Histogram of Average Distances to 4th Nearest Neighbor')
plt.xlabel('Average Distance')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Print the recommended epsilon
print("Recommended Epsilon:", recommended_epsilon)