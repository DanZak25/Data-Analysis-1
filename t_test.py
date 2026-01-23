import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv("dataset/2021-2022 Football Player Stats.csv")
midfielders = data[data['Pos'].str.startswith('MF')]
midfielders = midfielders[midfielders['Min'] >= 1500]

assists = midfielders["Assists"]
assist_25th = np.percentile(assists, 25)
assist_75th = np.percentile(assists, 75)

top_assists = midfielders[assists >= assist_75th]
bottom_assists = midfielders[assists <= assist_25th]

t_test_var = ["TouAtt3rd", "PPA", "Clr", "PresDef3rd"]
titles = [
    "Fig. 7 TouAtt3rd Comparison",
    "Fig. 8 PPA Comparison",
    "Fig. 9 Clr Comparison",
    "Fig. 10 PresDef3rd Comparison"
]

def perform_t_test(feature, group1, group2):
    t_stat, p_val = stats.ttest_ind(group1[feature], group2[feature])
    print(f"T-test for {feature}: t-statistic = {t_stat:.2f}, p-value = {p_val:.4f}")
    return t_stat, p_val

fig, axes = plt.subplots(2, 2, figsize=(14, 10))  

for (feature, title, ax) in zip(t_test_var, titles, axes.flatten()):
    t_stat, p_val = perform_t_test(feature, top_assists, bottom_assists)
    means = [top_assists[feature].mean(), bottom_assists[feature].mean()]
    stds = [top_assists[feature].std(), bottom_assists[feature].std()]
    
    ax.bar([1, 2], means, yerr=stds, color=["blue", "green"], alpha=0.7, capsize=10, edgecolor="black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Top 25%", "Bottom 25%"],fontsize=12)
    ax.set_title(title,fontsize=14)
    ax.set_ylabel(feature,fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.suptitle("Comparison of between 25th and 75th percentile in Assists",fontsize=16)
plt.tight_layout()
plt.show()
