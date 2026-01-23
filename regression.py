import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import RobustScaler

data = pd.read_csv("dataset/2021-2022 Football Player Stats.csv")

regression_vars = ["TouAtt3rd", "PPA", "Assists"]
titles = ["Fig. 3 TouAtt3rd vs Assists", "Fig. 4 PPA vs Assists"]

midfielders = data[data['Pos'].str.startswith('MF')]
midfielders = midfielders[midfielders['Min'] >= 1500]
regression_data = midfielders[regression_vars]

def linear_regression_plot(ax, x, y, data, title):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[x], data[y])
    print(f"Regression results for {title}:")
    print(f"Slope: {slope:.3f}, Intercept: {intercept:.1e}, R-value: {r_value:.3f}, P-value: {p_value:.1e}")
    
    ax.scatter(data[x], data[y], color='blue', s=50, label='Data Points')
    regression_line = slope * data[x] + intercept
    ax.plot(data[x], regression_line, color='red', label=f'Regression Line: y={slope:.2f}x + {intercept:.2f}')
    ax.text(0.95, 0.95, f"r = {r_value:.3f}", transform=ax.transAxes, 
            fontsize=12, ha='right', va='top', color='black', fontweight='bold')

    ax.set_title(title,fontsize=14)
    ax.set_xlabel(x,fontsize=12)
    ax.set_ylabel(y,fontsize=12)
    ax.legend()
    ax.grid(alpha=0.7, linestyle='--')

minute_thresholds = np.arange(1500, 2600, 200)
correlation_results = {var: [] for var in regression_vars[:2]}

for min_thresh in minute_thresholds:
    filtered_midfielders = data[data['Pos'].str.startswith('MF')]
    filtered_midfielders = filtered_midfielders[filtered_midfielders['Min'] >= min_thresh]
    
    for var in regression_vars[:2]: 
        if len(filtered_midfielders) > 2: 
            r_value, _ = stats.pearsonr(filtered_midfielders[var], filtered_midfielders["Assists"])
            correlation_results[var].append(r_value)
        else:
            correlation_results[var].append(np.nan) 

z_thresholds = np.arange(-2, 2.1, 0.1)  
z_correlation_results = {var: [] for var in regression_vars[:2]}

scaler = RobustScaler()
standardized_data = pd.DataFrame(scaler.fit_transform(regression_data), columns=regression_vars)

for var in regression_vars[:2]: 
    for z_thresh in z_thresholds:
        indices_below_threshold = np.where(standardized_data[var] < z_thresh)[0]
        if len(indices_below_threshold) > 2:  
            subset_x = standardized_data[var].iloc[indices_below_threshold]
            subset_y = standardized_data["Assists"].iloc[indices_below_threshold]
            r_value, _ = stats.pearsonr(subset_x, subset_y)
            z_correlation_results[var].append(r_value)
        else:
            z_correlation_results[var].append(np.nan)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for var, title, ax in zip(regression_vars[:2], titles, axes[0, :]):
    linear_regression_plot(ax, var, "Assists", regression_data, title)

for var, label in zip(regression_vars[:2], ["Correlation: TouAtt3rd vs Assists", "Correlation: PPA vs Assists"]):
    axes[1, 0].plot(minute_thresholds, correlation_results[var], marker="o", linestyle="-", label=label)

axes[1, 0].set_xlabel("Minimum Minutes Played (Threshold)",fontsize=12)
axes[1, 0].set_ylabel("Correlation Coefficient (r)",fontsize=12)
axes[1, 0].set_title("Fig. 5 Change in Correlation as Minimum Playing Time Increases",fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.7, linestyle="--")

for var, label in zip(regression_vars[:2], ["Correlation: TouAtt3rd vs Assists", "Correlation: PPA vs Assists"]):
    axes[1, 1].plot(z_thresholds, z_correlation_results[var], marker="o", linestyle="-", label=label)

axes[1, 1].set_xlabel("Z-Score Threshold",fontsize=12)
axes[1, 1].set_ylabel("Correlation Coefficient (r)",fontsize=12)
axes[1, 1].set_title("Fig. 6 Sensitivity Analysis: Correlation vs. Z-Score Threshold",fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.7, linestyle="--")

plt.tight_layout()
plt.show()
