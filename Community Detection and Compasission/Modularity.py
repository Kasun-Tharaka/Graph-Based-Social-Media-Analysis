import networkx as nx
import pandas as pd
from sklearn.cluster import SpectralClustering
import numpy as np
from community import community_louvain

# Load pre-built graph
G = nx.read_gexf('youtube_communities.gexf')

# Extract largest connected component
largest_cc = max(nx.connected_components(G), key=len)
H = G.subgraph(largest_cc)

# 1. Louvain Method (previously applied)
louvain_partition = {node: H.nodes[node]['community'] for node in H.nodes}
louvain_modularity = community_louvain.modularity(louvain_partition, H, weight='weight')

# 2. Spectral Clustering
adj_matrix = nx.adjacency_matrix(H, weight='weight').astype(float)

# Determine optimal cluster count (k) via eigenvalue gap
eigenvalues = np.linalg.eigvalsh(adj_matrix.todense())
eig_gaps = np.diff(eigenvalues[::-1])
optimal_k = np.argmax(eig_gaps) + 2  # Add 2 to offset indexing

# Apply spectral clustering
spectral = SpectralClustering(
    n_clusters=optimal_k,
    affinity='precomputed',
    random_state=42,
    assign_labels='discretize'
)
spectral_labels = spectral.fit_predict(adj_matrix)
spectral_partition = dict(zip(H.nodes, spectral_labels))
spectral_modularity = community_louvain.modularity(spectral_partition, H, weight='weight')

print(f"Louvain Modularity: {louvain_modularity:.4f}")
print(f"Spectral Clustering Modularity: {spectral_modularity:.4f}")
print(f"Optimal clusters (Spectral): {optimal_k}")