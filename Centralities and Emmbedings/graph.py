import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from community import community_louvain
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
from node2vec import Node2Vec

# 1. Load the comments data
df = pd.read_csv('new_krish_youtube_comments.csv')

# 2. Create interaction graph
G = nx.Graph()

# Add nodes (users) with their names
for author_id, author_name in zip(df['author_channel_id'], df['author']):
    if author_id and not G.has_node(author_id):
        G.add_node(author_id, name=author_name)

# Add edges based on interactions
for _, row in df.iterrows():
    if not row['author_channel_id'] or not row['parent_id']:
        continue
        
    parent_row = df[df['comment_id'] == row['parent_id']]
    if not parent_row.empty:
        parent_author = parent_row['author_channel_id'].values[0]
        if parent_author != row['author_channel_id']:  # Avoid self-loops
            # Add or update edge weight
            if G.has_edge(row['author_channel_id'], parent_author):
                G[row['author_channel_id']][parent_author]['weight'] += 1
            else:
                G.add_edge(row['author_channel_id'], parent_author, weight=1)

# 3. Community Detection using Louvain method
partition = community_louvain.best_partition(G, weight='weight')
nx.set_node_attributes(G, partition, 'community')

# 4. Analyze communities
community_counts = Counter(partition.values())
print(f"Detected {len(community_counts)} communities")
print("Community sizes:")
for comm_id, count in community_counts.most_common():
    print(f"Community {comm_id}: {count} members")

# 5. Visualize the graph with communities
plt.figure(figsize=(15, 12))

# Get largest connected component
components = sorted(nx.connected_components(G), key=len, reverse=True)
largest_component = G.subgraph(components[0])

# Position nodes using spring layout
pos = nx.spring_layout(largest_component, seed=42, k=0.15)

# Map communities to colors
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
node_colors = [cmap(partition[n]) for n in largest_component]

# Draw the graph
nx.draw_networkx_nodes(
    largest_component, 
    pos, 
    node_size=50,
    node_color=node_colors,
    alpha=0.8
)

nx.draw_networkx_edges(
    largest_component,
    pos,
    alpha=0.1,
    width=0.5
)

# Add labels for important nodes (high degree)
degrees = dict(largest_component.degree())
important_nodes = [n for n in largest_component if degrees[n] > np.percentile(list(degrees.values()), 90)]
labels = {n: G.nodes[n]['name'] for n in important_nodes}
nx.draw_networkx_labels(largest_component, pos, labels, font_size=8)

plt.title("YouTube Commenter Communities")
plt.axis('off')
plt.tight_layout()
plt.savefig('community_visualization.png', dpi=300)
plt.show()

# 6. Advanced visualization with node2vec embeddings
node2vec = Node2Vec(largest_component, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get embeddings
embeddings = np.array([model.wv[str(node)] for node in largest_component.nodes()])

# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot embeddings
plt.figure(figsize=(14, 10))
plt.scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1],
    c=[partition[n] for n in largest_component.nodes()],
    cmap='tab20',
    s=20,
    alpha=0.8
)

# Annotate some central nodes
central_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
for node, _ in central_nodes:
    idx = list(largest_component.nodes()).index(node)
    plt.annotate(
        G.nodes[node]['name'],
        (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

plt.title("Community Structure in t-SNE Projection Space")
plt.colorbar(label='Community ID')
plt.axis('off')
plt.savefig('community_embeddings.png', dpi=300)
plt.show()

# 7. Community analysis and characterization
def analyze_community(community_id):
    """Analyze and characterize a specific community"""
    members = [n for n in G.nodes if partition.get(n) == community_id]
    
    # Top members by degree
    community_degrees = sorted(
        [(n, G.degree(n)) for n in members],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Community subgraph
    subgraph = G.subgraph(members)
    
    # Calculate metrics
    return {
        'id': community_id,
        'size': len(members),
        'density': nx.density(subgraph),
        'avg_clustering': nx.average_clustering(subgraph),
        'central_members': [
            (G.nodes[n]['name'], deg) for n, deg in community_degrees
        ],
        'topics': extract_community_topics(members)
    }

def extract_community_topics(members):
    """Extract common topics from community comments"""
    member_comments = df[df['author_id'].isin(members)]['text']
    all_text = ' '.join(member_comments.astype(str))
    
    # Simple word frequency analysis (in real use, apply NLP preprocessing)
    words = pd.Series(all_text.lower().split())
    word_counts = words.value_counts().head(10)
    
    # Remove common stop words
    stop_words = {'the', 'and', 'to', 'of', 'a', 'i', 'is', 'in', 'it', 'this', 'that', 'for'}
    return [word for word in word_counts.index if word not in stop_words][:5]

# Analyze top communities
community_results = []
for comm_id, count in community_counts.most_common(5):
    comm_data = analyze_community(comm_id)
    community_results.append(comm_data)
    print(f"\nCommunity {comm_id} (Size: {count}):")
    print(f"Density: {comm_data['density']:.4f}, Clustering: {comm_data['avg_clustering']:.4f}")
    print("Central Members:")
    for name, deg in comm_data['central_members']:
        print(f"  - {name} (Degree: {deg})")
    print(f"Common Topics: {', '.join(comm_data['topics'])}")

# 8. Save community analysis to CSV
comm_df = pd.DataFrame(community_results)
comm_df.to_csv('community_analysis.csv', index=False)

# 9. Save graph with communities for further analysis
nx.write_gexf(G, 'youtube_communities.gexf')