import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

choice = input("You can choose between [S]catter plot or [D]endogram\n")

if choice in ['S', 's']:
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster.fit_predict(my_plot)
    #print(cluster.labels_)
    plt.scatter(my_plot[:, 0], my_plot[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.xlabel("price")
    plt.ylabel("availability")
    plt.show()
    input("Press enter to continue\n")
elif choice in ['D', 'd']:
    my_link = linkage(my_plot, 'ward')
    plt.figure(figsize=(25, 10))
    dendrogram(my_link, leaf_rotation=90, leaf_font_size=8)
    plt.show()
    input("Press enter to continue\n")
else:
    print("Sorry I don't recognise this input\n End of program\n")
print("Program finished, time to drink coffee.\"")