import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("dataset/2021-2022 Football Player Stats.csv")
midfielders = data[data['Pos'].str.startswith('MF')]
midfielders = midfielders[midfielders['Min'] >= 1500]

kmean_variables = midfielders[["Assists", "TouAtt3rd", "PPA"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(kmean_variables)
midfielders['Cluster'] = clusters

def extract_top_3_midfielders(df):
    top_players = []
    for squad, group in df.groupby("Squad"):
        top_3 = group.nlargest(3, "Min") 
        top_players.append(top_3.head(3))  
    return pd.concat(top_players, ignore_index=True)

top_3_midfielders_per_squad = extract_top_3_midfielders(midfielders)
cluster_counts = top_3_midfielders_per_squad["Cluster"].value_counts().sort_index()
selected_stats = ["Assists", "TouAtt3rd", "PPA", "RecProg", "Clr", "PresDef3rd", "Int"]
mean_values_per_cluster = top_3_midfielders_per_squad.groupby("Cluster")[selected_stats].mean()

comp_cluster_counts = top_3_midfielders_per_squad.groupby(["Comp", "Cluster"]).size().unstack(fill_value=0)
comp_cluster_counts.rename(columns={0: "Cluster 0 Count", 1: "Cluster 1 Count", 2: "Cluster 2 Count"}, inplace=True)

plt.figure(figsize=(10, 6))

leagues = comp_cluster_counts.index
cluster_0_counts = comp_cluster_counts["Cluster 0 Count"]
cluster_1_counts = comp_cluster_counts["Cluster 1 Count"]
cluster_2_counts = comp_cluster_counts["Cluster 2 Count"]

plt.bar(leagues, cluster_0_counts, label="Cluster 0", color="blue", alpha=0.7)
plt.bar(leagues, cluster_1_counts, bottom=cluster_0_counts, label="Cluster 1", color="green", alpha=0.7)
plt.bar(leagues, cluster_2_counts, bottom=cluster_0_counts + cluster_1_counts, label="Cluster 2", color="red", alpha=0.7)

plt.xlabel("League (Comp)", fontsize=12)
plt.ylabel("Number of Players", fontsize=12)
plt.title("Fig. 16 Distribution of Players in Clusters Across Leagues", fontsize=14)
plt.legend(title="Cluster")

plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
