import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("delhi_air.csv")

X = df[['PM2.5', 'PM10', 'CO', 'Ozone', 'AQI']]
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)

# ========== ELBOW METHOD ==========
inertia = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(x_scaled)
    inertia.append(km.inertia_)

plt.figure()
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")  # FIX: Added axis labels
plt.ylabel("Inertia")
plt.plot(range(1, 10), inertia, marker='o')
plt.show()

# ========== KMEANS CLUSTERING ==========
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(x_scaled)

print("KMeans Cluster Distribution:")
print(df['cluster'].value_counts())

plt.figure()
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel("PM2.5 (scaled)")
plt.ylabel("PM10 (scaled)")
plt.title("KMeans Clusters")
plt.show()

# ========== HIERARCHICAL CLUSTERING ==========
linked = linkage(x_scaled, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

agg = AgglomerativeClustering(n_clusters=3)
df['agg_cluster'] = agg.fit_predict(x_scaled)
print("\nAgglomerative Cluster Distribution:")
print(df['agg_cluster'].value_counts())

plt.figure()
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=df['agg_cluster'], cmap='viridis')
plt.xlabel("PM2.5 (scaled)")
plt.ylabel("PM10 (scaled)")
plt.title("Agglomerative Clusters")
plt.show()

# ========== EVALUATION ==========
print("\nKMeans Silhouette Score:", silhouette_score(x_scaled, df['cluster']))
print("Agglomerative Silhouette Score:", silhouette_score(x_scaled, df['agg_cluster']))

# ========== INFERENCE ==========
"""
INFERENCE:
1. Elbow Method helps find optimal number of clusters (bend in the curve).
2. KMeans partitions data into k clusters based on distance to centroids.
3. Agglomerative Clustering merges closest points bottom-up (hierarchical).
4. Dendrogram visualizes the merging process and helps decide number of clusters.
5. Silhouette Score measures cluster quality (-1 to 1, higher = better).
6. MinMaxScaler is essential — clustering is distance-based, so features must be on same scale.
"""
