import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("dataset/2021-2022 Football Player Stats.csv")
midfielders = data[data['Pos'].str.startswith('MF')]
midfielders = midfielders[midfielders['Min'] >= 1500]

kmean_variables = midfielders[["Assists", "TouAtt3rd", "PPA"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(kmean_variables)
midfielders['Cluster'] = clusters

silhouette_avg = silhouette_score(kmean_variables, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')

for cluster in np.unique(clusters):
    cluster_points = kmean_variables[clusters == cluster] 
    ax1.scatter(
        cluster_points["Assists"],  
        cluster_points["TouAtt3rd"],  
        cluster_points["PPA"],  
        label=f"Cluster {cluster}",
        s=50
    )

ax1.set_xlabel("Assists",fontsize=12)
ax1.set_ylabel("TouAtt3rd",fontsize=12)
ax1.set_zlabel("PPA", labelpad=2, rotation=90)
ax1.set_title("Fig. 14 Player Clustering(K=3)",fontsize=14)
ax1.legend()

k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_k.fit(kmean_variables)
    labels_k = kmeans_k.labels_
    
    sscore = silhouette_score(kmean_variables, labels_k)
    silhouette_scores.append(sscore)

ax2 = fig.add_subplot(122)
ax2.plot(k_values, silhouette_scores, marker='o', linestyle='-')
ax2.set_xlabel("Number of Clusters (K)",fontsize=12)
ax2.set_ylabel("Silhouette Score",fontsize=12)
ax2.set_title("Fig. 15 Silhouette Score vs. Number of Clusters (K)",fontsize=14)

plt.tight_layout()
plt.show()
