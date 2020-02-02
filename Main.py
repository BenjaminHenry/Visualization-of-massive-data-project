import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from os import listdir
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Chargement du fichier CSV
my_csvs = []
my_files = listdir(os.getcwd())
for my_files in my_files:
    if my_files.endswith(".csv"):
        my_csvs.append(my_files)
if len(my_csvs) == 1:
    print("1 csv file detected :", my_csvs)
    print("Using :", my_csvs)
elif len(my_csvs) > 1:
    print("please select a file between these .csv found :", my_csvs)
    my_chosenfile = input()
elif len(my_csvs) < 1:
    print("Sorry no cvs files found exiting program")
    exit(0)
if my_chosenfile.casefold() in str(my_csvs).strip('[]').casefold():
    print("You selected :", my_chosenfile)
else:
    print("Sorry, wrong selection, exiting program")
    exit(0)
data = pd.read_csv(my_chosenfile, index_col=0)


#selection des colonne de type int ou float pour proposer le choix à l'utilisateur
my_type = list(data.select_dtypes(include=['int64']).columns)
my_type += (list(data.select_dtypes(include=['float64']).columns))

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

#Filtrage des valeurs nulles
data = data.loc[data[first_input] != 0]
data = data.loc[data[second_input] != 0]
column1 = np.asarray(data[first_input])
column2 = np.asarray(data[second_input])

#creation de la liste de valeur récupéré
my_plot = np.column_stack((column1, column2))

# Selection de valeurs randoms dans les colonnes à observer
if (len(column1) < 5000):
    randomRow = np.random.randint(len(column1), size=len(column1))
else:
    randomRow = np.random.randint(len(column1), size=5000)
my_plot = my_plot[randomRow, :]

choice = input("You can choose between [S]catter plot or [D]endogram\n")

if choice in ['S', 's']:
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(my_plot)
    plt.scatter(my_plot[:, 0], my_plot[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.xlabel(first_input)
    plt.ylabel(second_input)
    plt.show()
elif choice in ['D', 'd']:
    my_link = linkage(my_plot, 'ward')
    plt.figure(figsize=(25, 10))
    dendrogram(my_link, leaf_rotation=90, leaf_font_size=8)
    plt.xlabel(first_input)
    plt.ylabel(second_input)
    plt.show()
else:
    print("Sorry I don't recognise this input\n End of program")

#start of test for predictive model of one column as a function of another column
#clf = svm.SVC(gamma=0.001, C=100)
#clf.fit(my_plot.data[:-1], my_plot.target[:-1])
#clf.predict(my_plot.data[:-1])

#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(my_plot, my_plot)
#pred = neigh.predict(my_plot)
#print("KNeighbors accuracy score : ", accuracy_score(my_plot, pred))

print("Program finished, time to drink coffee.")
