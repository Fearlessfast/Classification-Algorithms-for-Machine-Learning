import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

plt.interactive(True)
dbscan = pd.read_csv("AlcoholicBeveragesAnalysis.csv")
# Dropping the Beverages column from the data
dbscan = dbscan.drop('Beverages', axis=1)

# Handling the missing values
dbscan.fillna(method='ffill', inplace=True)

print(dbscan.head())
print("***********************************************")
print("***********************************************")

# Scaling the data to bring all the attributes to a comparable level
Scaler = StandardScaler()
dbscan_scaled = Scaler.fit_transform(dbscan)
dbscan_normalized = normalize(dbscan_scaled)

# Converting the numpy array into a pandas DataFrame
dbscan_normalized = pd.DataFrame(dbscan_scaled)

pca = PCA(n_components = 2)
dbscan_principal = pca.fit_transform(dbscan_normalized)
dbscan_principal = pd.DataFrame(dbscan_principal)
dbscan_principal.columns = ['P1', 'P2']
print(dbscan_principal.head())

# Numpy array of all the cluster labels assigned to each data point
db_default = DBSCAN(eps = 0.0375, min_samples = 3).fit(dbscan_principal)
labels = db_default.labels_

# Building the label to colour mapping
colours = {}
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'

# Building the colour vector for each data point
cvec = [colours[label] for label in labels]

# For the construction of the legend of the plot
r = plt.scatter(dbscan_principal['P1'], dbscan_principal['P2'], color='r');
g = plt.scatter(dbscan_principal['P1'], dbscan_principal['P2'], color='g');
b = plt.scatter(dbscan_principal['P1'], dbscan_principal['P2'], color='b');
k = plt.scatter(dbscan_principal['P1'], dbscan_principal['P2'], color='k');

# Plotting P1 on the X-Axis and P2 on the Y-Axis
# according to the colour vector defined
plt.figure(figsize=(9, 9))
plt.scatter(dbscan_principal['P1'], dbscan_principal['P2'], c=cvec)

# Building the legend
plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))

plt.show()