from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plotnine import ggplot, aes, geom_point
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

plt.interactive(True)
gmm = pd.read_csv("AlcoholicBeveragesAnalysis.csv")

print(gmm.head())
print("***********************************************")
print("***********************************************")

Beverages = LabelEncoder()
gmm['Beverages_new'] = Beverages.fit_transform(gmm['Beverages'])

print(gmm.head())
print("***********************************************")
print("***********************************************")

gmm_n = gmm.drop(['Beverages'], axis='columns')
print(gmm_n)

features = ["Calories", "Serving Size(ml)"]

inputs = gmm_n[features]
SS = StandardScaler()

inputs[features] = SS.fit_tranform(inputs)

EM = GaussianMixture(n_components = 3)
EM.fit(inputs)

cluster = EM.predict(inputs)


print(cluster)

print("***********************************************")
print("***********************************************")

cluster_p = EM.predict_proba(inputs)

print(cluster_p)

print("***********************************************")
print("***********************************************")

print("silhouette: ", silhouette_score(inputs, cluster))

print("***********************************************")
print("***********************************************")

inputs["cluster"] = cluster

ggplot(inputs, aes(x = "Calories", y = "Serving Size(ml)", color = "cluster")) + geom_point()
