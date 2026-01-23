import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

data = pd.read_csv("dataset/2021-2022 Football Player Stats.csv")
midfielders = data[data['Pos'].str.startswith('MF')]
midfielders = midfielders[midfielders['Min'] >= 1500]

assists = midfielders["Assists"]
assist_25th = np.percentile(assists, 25)
assist_75th = np.percentile(assists, 75)
filtered_midfielders = midfielders[(assists <= assist_25th) | (assists >= assist_75th)]

ppa_filtered = filtered_midfielders["PPA"].values
touatt_filtered = filtered_midfielders["TouAtt3rd"].values
features_filtered = np.column_stack((ppa_filtered, touatt_filtered))

scaler = RobustScaler(with_centering=True, with_scaling=True).fit(features_filtered)
features_scaled = scaler.transform(features_filtered)

ppa_scaled, touatt_scaled = features_scaled[:, 0], features_scaled[:, 1]

fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

axs[0].hist(ppa_filtered, bins=20, color='blue', alpha=0.7, label='Original PPA', edgecolor='black')
axs[0].hist(touatt_filtered, bins=20, color='green', alpha=0.7, label='Original TouAtt3rd', edgecolor='black')
axs[0].set_title("Figure 11. Original Distributions",fontsize=14)
axs[0].set_xlabel("Values(Scaled)",fontsize=12)
axs[0].set_ylabel("Frequency",fontsize=12)
axs[0].legend()

axs[1].hist(ppa_scaled, bins=20, color='blue', alpha=0.7, label='Robust Scaled PPA', edgecolor='black')
axs[1].hist(touatt_scaled, bins=20, color='green', alpha=0.7, label='Robust Scaled TouAtt3rd', edgecolor='black')
axs[1].set_title("Figure 12. Transformed Distributions",fontsize=14)
axs[1].set_xlabel("Values(Scaled)",fontsize=12)
axs[1].set_ylabel("Frequency",fontsize=12)
axs[1].legend()

plt.suptitle("Comparison of Original and Robust Scaled Feature Distributions",fontsize=16)
plt.show()
