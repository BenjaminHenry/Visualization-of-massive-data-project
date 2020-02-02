import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

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

# PART 2 : Quantitive analysis

column1 = np.asarray(data['price'])
column2 = np.asarray(data['minimum_nights'])

# Filtrage des valeurs nulles
column1 = list(filter(lambda x:x != 0, column1))
column2 = list(filter(lambda x:x != 0, column2))

xlim_left = min(column1)
xlim_right = max(column1)
ylim_top = max(column2)
ylim_bottom = min(column2)

"""
randomly select training set and test set from the dataset
"""

nb_points = len(column1)
nb_training_points = int(0.7 * nb_points)
training_indexes = random.sample(range(nb_points), nb_training_points)
test_indexes = [index for index in range(nb_points)
                if index not in training_indexes]

x_train = [column1[i] for i in training_indexes]
y_train = [column2[i] for i in training_indexes]
x_test = [column1[i] for i in test_indexes]
y_test = [column2[i] for i in test_indexes]


def plot_polynom_sample(polynom, x_train, y_train):
    """
        Plot the result of fitting the polynom
        to the training set
    """
    degree = len(polynom)-1
    title = f"Polynomial fit on training set, degree={degree}"
    filename = f"Fit_degree_{degree}.pdf"
    x_plot = np.linspace(xlim_left, xlim_right, 500)
    plt.plot(x_train,
             y_train,
             'o',
             x_test,
             y_test,
             'x',
             x_plot,
             np.polyval(polynom, x_plot[::-1]), '-')
    plt.legend(['training set', 'test set', 'model'], loc='best')
    plt.xlabel('price')
    plt.ylabel('minimum_nights')
    plt.xlim(xlim_left, xlim_right)
    plt.ylim(ylim_bottom, ylim_top)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def compute_test_error(polynom, x_test, y_test):
    """
        Evaluate the quality of out model on the test set.
        We compute the Mean Square Error.
    """
    # compare prediction to ground truth
    errors = np.polyval(polynom, x_test) - y_test
    square_errors = [error**2 for error in errors]
    total_error = sum(square_errors)
    mean_square_error = total_error/len(square_errors)
    return mean_square_error


def compute_training_error(polynom, x_train, y_train):
    """
        Evaluate the quality of out model on the training set.
        We compute the Mean Square Error.
    """
    # compare prediction to ground truth
    errors = np.polyval(polynom, x_train) - y_train
    square_errors = [error**2 for error in errors]
    total_error = sum(square_errors)
    mean_square_error = total_error/len(square_errors)
    return mean_square_error

# degré d'entraînement, plus élevé = plus précis, trop élevé = overfitting
degree = 8

poly = np.polyfit(x_train, y_train, degree)
print(f"mean square error on training set: {compute_training_error(poly, x_train, y_train)}")
print(f"mean square error on test set: {compute_test_error(poly, x_test, y_test)}")
plot_polynom_sample(poly, x_train, y_train)

print("Program finished, time to drink coffee.")