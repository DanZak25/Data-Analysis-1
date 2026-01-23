Tradeimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/2021-2022 Football Player Stats.csv")
midfielders = data[data['Pos'].str.startswith('MF')]
midfielders = midfielders[midfielders['Min'] >= 1500]

independent_vars = ['Goals', 'PasTotCmp', 'PasShoCmp', 'PasMedCmp', 'PasLonCmp', 'Crs', 'Assists', 'PPA', 
                    'CrsPA', 'PasProg', 'PasAtt', 'TB', 'PasPress', 'Sw', 'PasCmp', 'Tkl', 'TklWon', 
                    'TklDef3rd', 'TklMid3rd', 'TklAtt3rd', 'PresDef3rd', 'PresMid3rd', 'PresAtt3rd', 
                    'Blocks', 'Int', 'Clr', 'TouDef3rd', 'TouMid3rd', 'TouAtt3rd', 'DriSucc', 'CarProg', 
                    'Car3rd', 'CPA', 'RecProg', 'Recov', 'AerWon']
dependent_vars = ["Assists"]

correlation = midfielders[independent_vars].corrwith(midfielders[dependent_vars[0]]).to_frame(name="Assists")

correlation = correlation.drop(index=["Assists"])


top_3_positive = correlation.sort_values("Assists", ascending=False).head(3)
top_3_negative = correlation.sort_values("Assists", ascending=True).head(3)

def plot_correlation(ax, correlation, title, color_map):
    correlation.plot(kind="bar", colormap=color_map, edgecolor="black", width=0.8, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Variables", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_correlation(axes[0], top_3_positive, "Fig. 1 Top 3 Most Positively Correlated Features with Assists", "Oranges")
plot_correlation(axes[1], top_3_negative, "Fig. 2 Top 3 Most Negatively Correlated Features with Assists", "Blues")

plt.tight_layout()
plt.show()


