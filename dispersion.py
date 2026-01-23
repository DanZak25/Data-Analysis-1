import numpy as np
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

top_3_features = top_3_positive.index.append(top_3_negative.index).append(pd.Index(['Assists']))

dispersion_metrics = pd.DataFrame(columns=[
    'Min', 'Max', 'Range', '25th Percentile', '75th Percentile', 'Interquartile Range', 'Standard Deviation', 'Variance'])

for feature in top_3_features:
    feature_data = midfielders[feature]
    min_val = feature_data.min()
    max_val = feature_data.max()
    range_val = max_val - min_val
    percentile_25 = np.percentile(feature_data, 25)
    percentile_75 = np.percentile(feature_data, 75)
    iqr = percentile_75 - percentile_25
    std_dev = feature_data.std()
    variance = feature_data.var()
    dispersion_metrics.loc[feature] = [min_val, max_val, range_val, percentile_25, percentile_75, iqr, std_dev, variance]

if "90s" in midfielders.columns:
    feature_data = midfielders["90s"]
    min_val = feature_data.min()
    max_val = feature_data.max()
    range_val = max_val - min_val
    percentile_25 = np.percentile(feature_data, 25)
    percentile_75 = np.percentile(feature_data, 75)
    iqr = percentile_75 - percentile_25
    std_dev = feature_data.std()
    variance = feature_data.var()
    dispersion_metrics.loc["90s"] = [min_val, max_val, range_val, percentile_25, percentile_75, iqr, std_dev, variance]

