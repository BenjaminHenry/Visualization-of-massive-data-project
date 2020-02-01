import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Chargement du fichier CSV
data = pd.read_csv("AB_NYC_2019.csv", index_col=0)

#selection des colonne de type int pour proposer le choix à l'utilisateur
my_type = list(data.select_dtypes(include=['int64']).columns)

# Selection des 2 colonnes à observer
print("Please choose a first column between these : ")
print(*my_type, sep = ", ")
first_input = input()
if first_input in my_type:
    my_type.remove(first_input)
    print("Please choose a second column between these : ")
    print(*my_type, sep = ", ")
    second_input = input()
else:
    print("wrong input sorry, program restarting")
    exit(0)
if second_input in my_type:
    print("Thanks for your choices")
else:
    print("wrong input sorry, program restarting")
    exit(0)
column1 = np.asarray(data[first_input])
column2 = np.asarray(data[second_input])

# Filtrage des valeurs nulles
column1 = list(filter(lambda x:x > 0, column1))
column2 = list(filter(lambda x:x > 0, column2))

# Recoupage des colonnes pour avoir la même taille
if len(column1) > len(column2):
    column1 = column1[:len(column2)]
elif len(column2) > len(column1):
    column2 = column2[:len(column1)]

my_plot = np.column_stack((column1, column2))

# Selection de valeurs randoms dans les colonnes à observer
randomRow = np.random.randint(len(column1), size=100)
my_plot = my_plot[randomRow, :]

choice = input("You can choose between [S]catter plot or [D]endogram\n")

if choice in ['S', 's']:
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster.fit_predict(my_plot)
    plt.scatter(my_plot[:, 0], my_plot[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.xlabel(first_input)
    plt.ylabel(second_input)
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