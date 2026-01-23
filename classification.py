import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

data = pd.read_csv("dataset/2021-2022 Football Player Stats.csv")

midfielders = data[data['Pos'].str.startswith('MF')]
midfielders = midfielders[midfielders['Min'] >= 1500]
assists = midfielders["Assists"]
assist_25th = np.percentile(assists, 25)
assist_75th = np.percentile(assists, 75)
top_assists = midfielders[assists >= assist_75th]
bottom_assists = midfielders[assists <= assist_25th]

mlp_variables = midfielders[["PPA", "TouAtt3rd"]]
labels = np.where(assists <= assist_25th, '25th', np.where(assists >= assist_75th, '75th', None))
valid_rows = labels != None
mlp_variables = mlp_variables[valid_rows]
labels = np.array(labels)[valid_rows]
label_map = {'25th': 0, '75th': 1}
labels_numeric = np.array([label_map[label] for label in labels])

def run_single_iteration_with_robust_scaling(x, labels, test_size=0.2, random_state=1, alpha=0.1, max_iter=500):
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=test_size, random_state=random_state, stratify=labels)

    transformer = RobustScaler(with_centering=True, with_scaling=True).fit(x_train)
    x_train_scaled = transformer.transform(x_train)
    x_test_scaled = transformer.transform(x_test)

    mlp = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(40, 20), random_state=random_state, max_iter=max_iter)
    mlp.fit(x_train_scaled, labels_train)

    labels_test_pred = mlp.predict(x_test_scaled)
    cr = accuracy_score(labels_test, labels_test_pred)

    return mlp, transformer, x_train_scaled, x_test_scaled, labels_train, labels_test, cr  

def report_cr_results(cr, header=""):   
    cr = np.asarray(cr)
    print(header, ':')
    print(f'   Average classification rate (CR) = {np.around(cr.mean(), 3)}')
    print()

def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
    nlabels = np.unique(labels).size
    colors = plt.cm.viridis(np.linspace(0, 1, nlabels)) if (colors is None) else colors
    ax = plt.gca() if (ax is None) else ax
    xmin, xmax = x.min(axis=0), x.max(axis=0)
    
    xp, yp = np.meshgrid(np.linspace(xmin[0], xmax[0], n), np.linspace(xmin[1], xmax[1], n))
    x_flat = np.vstack([xp.flatten(), yp.flatten()]).T
    
    labels_pred = classifier.predict(x_flat)
    labels_pred_reshaped = np.reshape(labels_pred, xp.shape)
    
    cmap = ListedColormap(colors)
    for i, label in enumerate(np.unique(labels)):
        xx = x[labels == label]
        ax.scatter(xx[:, 0], xx[:, 1], color=colors[i], s=marker_size, alpha=marker_alpha, label=f'Label = {label}')
    plt.pcolormesh(xp, yp, labels_pred_reshaped, cmap=cmap, alpha=alpha)
    ax.set_xlabel('PPA(Scaled)',fontsize=12)
    ax.set_ylabel('TouAtt3rd(Scaled)',fontsize=12)
    ax.set_title('Fig. 13 Multi-Layer Perceptron Classification on 25th and 75th percentile',fontsize=14)
    ax.axis('equal')
    ax.legend()

niter = 10
max_iter = 500
kwargs = {'test_size': 0.2, 'alpha': 0.1, 'max_iter': max_iter}

mlp_models = [run_single_iteration_with_robust_scaling(mlp_variables, labels_numeric, random_state=i, **kwargs) for i in range(niter)]

cr_with_scaling = [result[6] for result in mlp_models]

report_cr_results(cr_with_scaling, header="Machine Learning Results for Midfielders in the 25th and 75th Assists Percentile (With RobustScaler)")

mlp_model, scaler, x_train_scaled, x_test_scaled, labels_train, labels_test, _ = mlp_models[0]

plt.figure(figsize=(8, 6))
plot_decision_surface(mlp_model, x_test_scaled, labels_test, colors=['b', 'r'])
plt.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=['b' if label == 0 else 'r' for label in labels_test], marker='o')
plt.show()


