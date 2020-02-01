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
from scipy.cluster.hierarchy import dendrogram, linkage

# Chargement du fichier CSV
data = pd.read_csv("AB_NYC_2019.csv", index_col=0)

# Selection des 2 colonnes à observer
my_avail = np.asarray(data['availability_365'])
my_price = np.asarray(data['price'])

# Filtrage des valeurs nulles
my_avail = list(filter(lambda x:x > 0, my_avail))

# Recoupage de la deuxieme colonne pour avoir la même taille
my_price = my_price[:len(my_avail)]

my_plot = np.column_stack((my_price, my_avail))

# Selection de valeurs randoms dans les colonnes à observer
randomRow = np.random.randint(len(my_avail), size=100)
my_plot = my_plot[randomRow, :]

#print(my_plot)

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(my_plot)
#print(cluster.labels_)
plt.scatter(my_plot[:, 0], my_plot[:, 1], c=cluster.labels_, cmap='rainbow')
plt.xlabel("price")
plt.ylabel("availability")
plt.show()
input()

my_link = linkage(my_plot, 'ward')
plt.figure(figsize=(25, 10))
dendrogram(my_link, leaf_rotation=90, leaf_font_size=8)
plt.show()
input()