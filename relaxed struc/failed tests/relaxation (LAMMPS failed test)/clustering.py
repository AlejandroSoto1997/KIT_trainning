from sklearn.cluster import DBSCAN
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Your sequences
sequences = [
    "CGATTGACTCTCCACGCTGTCCCTAACCATGACCGTCGAAG",
    "CGATTGACTCTCCTTCGACGGTCATGTACTAGATCAGAGG",
    "CGATTGACTCTCCCTCTGATCTAGTAGTTAGGACAGCGTG"
]

# Function to calculate pairwise identity percentage
def calculate_identity_percentage(seq1, seq2):
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return (matches / max(len(seq1), len(seq2))) * 100

# Construct a similarity matrix
similarity_matrix = np.zeros((len(sequences), len(sequences)))

for i in range(len(sequences)):
    for j in range(i + 1, len(sequences)):
        similarity_matrix[i, j] = calculate_identity_percentage(sequences[i], sequences[j])
        similarity_matrix[j, i] = similarity_matrix[i, j]

# Perform hierarchical clustering
Z = linkage(similarity_matrix, method='complete')

# DBSCAN clustering
epsilon = 0.5886296509062473  
min_samples = 1 
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

# Fit the DBSCAN model to the similarity matrix
labels = dbscan.fit_predict(similarity_matrix)

# Plot the dendrogram
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
dendrogram(Z, labels=["sequence1", "sequence2", "sequence3"])
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sequences')
plt.ylabel('Distance')

# Display DBSCAN clustering results
print("DBSCAN Cluster Labels:", labels)

# Plot the DBSCAN clustering results
plt.subplot(1, 2, 2)
plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering')
plt.xlabel('Sequences')
plt.ylabel('Cluster Labels')

plt.tight_layout()
plt.show()

# Visualize the similarity matrix
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='viridis', interpolation='none')
plt.colorbar(label='Similarity (%)')
plt.title('Pairwise Similarity Matrix')
plt.xticks(range(len(sequences)), ["sequence1", "sequence2", "sequence3"], rotation=45)
plt.yticks(range(len(sequences)), ["sequence1", "sequence2", "sequence3"])
plt.show()

# Visualize the similarity matrix with sequence labels
# Visualize the similarity matrix with sequence labels inside squares
# Visualize the similarity matrix with sequence labels

#sequence1 = "CGATTGACTCTCCA--CG-CTG-TCCCTAACCATG-ACC--G-TC-GAAG-"
#sequence2 = "CGATTGACTCTCC-TTCGAC-GGTC---A----TGTAC-TAGATCAGA-GG"
#sequence3 = "CGATTGACTCTCCCT-CTGA---TC-TAGTA-GTTAGGA-CAG-CGTG"