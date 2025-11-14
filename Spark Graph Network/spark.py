from pyspark.sql import SparkSession
from graphframes import GraphFrame
import time
import matplotlib.pyplot as plt
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("YouTubeCommentGraph") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12") \
    .getOrCreate()

spark.sparkContext.setCheckpointDir("/tmp/checkpoints")


df = spark.read.csv("youtube_comments.csv", header=True, inferSchema=True)

comments = df.select("comment_id", "user_id").distinct()

replies = df.filter(df.is_reply == True) \
    .select(
        df.user_id.alias("src_user_id"),
        df.parent_comment_id.alias("dst_comment_id") 
    )

replies_aliased = replies.alias("r")
comments_aliased = comments.alias("c")

edges = replies_aliased.join(comments_aliased, col("r.dst_comment_id") == col("c.comment_id")) \
    .select(
        col("r.src_user_id").alias("src"),
        col("c.user_id").alias("dst")
    )

nodes = df.select("user_id").distinct() \
    .withColumnRenamed("user_id", "id")

g = GraphFrame(nodes, edges)

def run_algorithms(graph, fraction):
    results = {}

    start = time.time()
    pr = graph.pageRank(resetProbability=0.15, maxIter=10)
    pr.vertices.count()
    results["pagerank_time"] = time.time() - start

    start = time.time()
    cc = graph.connectedComponents()
    cc.count()
    results["cc_time"] = time.time() - start

    results["nodes"] = graph.vertices.count()
    results["edges"] = graph.edges.count()
    results["fraction"] = fraction
    return results

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
metrics = []

for frac in fractions:
    sampled_edges = edges.sample(frac, seed=42)
    sampled_graph = GraphFrame(nodes, sampled_edges)
    metrics.append(run_algorithms(sampled_graph, frac))


for m in metrics:
    print(f"Fraction: {m['fraction']:.1f} | "
          f"Nodes: {m['nodes']} | Edges: {m['edges']} | "
          f"PageRank: {m['pagerank_time']:.1f}s | "
          f"CC: {m['cc_time']:.1f}s")

plt.figure(figsize=(10,6))
plt.plot([m['edges'] for m in metrics], [m['pagerank_time'] for m in metrics], 'o-', label='PageRank')
plt.plot([m['edges'] for m in metrics], [m['cc_time'] for m in metrics], 's-', label='Connected Components')
plt.xlabel('Number of Edges')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.savefig('scalability.png')