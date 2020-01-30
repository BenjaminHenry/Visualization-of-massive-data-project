import matplotlib
import scipy
import os
import numpy as np
import pandas as pd
import ipdb
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import csv

from sklearn.cluster import AgglomerativeClustering
# Chargement du fichier CSV
data = pd.read_csv("AB_NYC_2019.csv", index_col=0)

# Selection des 2 colonnes à observer
my_avail = np.asarray(data['minimum_nights'])
my_price = np.asarray(data['price'])
my_plot = np.column_stack((my_price, my_avail))

# Selection de valeurs randoms dans les colonnes à observer
randomRow = np.random.randint(40000, size=50)
my_plot = my_plot[randomRow, :]

#print(my_plot)

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(my_plot)
#print(cluster.labels_)
plt.scatter(my_plot[:, 0], my_plot[:, 1], c=cluster.labels_, cmap='rainbow')
plt.show()
input()