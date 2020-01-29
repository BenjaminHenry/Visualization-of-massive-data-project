import matplotlib
import scipy
import os
import numpy as np
import pandas as pd
import ipdb
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

'''
column_1 = np.random.uniform(3, 5, 10)

noise_2 = np.random.normal(0, 0.01, 10)
column_2 = 4*column_1+noise_2

noise_3 = np.random.normal(0, 0.1, 10)
column_3 = 50 - 2*column_2+noise_3

noise_4 = np.random.normal(0, 5, 10)
column_4 = 10+column_1+noise_4

data = np.column_stack((column_1, column_2))
data = np.column_stack((data, column_3))
data = np.column_stack((data, column_4))

np.save("data", data)

data = np.load("data.npy")

fig = plt.figure()


for point_index in range(data.shape[0]):
    pass


labels = ["Var 1", "Var 2", "Var 3", "Var 4"]
plt.title("parallel coordinate plot")
plt.xticks([0, 1, 2, 3], labels=labels)
plt.savefig("parallel_coordinates.pdf")
plt.close()

labels={"species_id": "Species", "sepal_width": "Sepal Width", "sepal_length": "Sepal Length", "petal_width": "Petal Width", "petal_length": "Petal Length", }

df = px.data.iris()
fig = px.parallel_coordinates(df,
                              color="species_id",
                              labels=labels,
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
fig.show()

#-------------------------------------------------------------------------------------------

labels = "Île-de-France", "Rhône-Alpes", "Provence Alpes\nCôte d'Azur", "Nord Pas de Calais"
areas = [12011, 43698, 31400, 12414]  # in km2
population = [11577, 6058, 4818, 4048]  # in million
colors = ['darkcyan', 'lightgreen', 'lightsteelblue', 'lightskyblue']
explode = (1, 0, 0, 0)

plt.pie(areas,
        # explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        )
plt.axis('equal')
plt.savefig("france_region_areas.pdf")
plt.close()

plt.pie(population,
        # explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        )

plt.axis('equal')
plt.savefig("france_region_population.pdf")
plt.close()

sns.set(style="ticks")

# load the dataset
df = sns.load_dataset("iris")

# pairplot is the name of the seaborn function
# to plot the scatter matrix
sns.pairplot(df, hue="species")

title = 'Scatter matrix of the iris dataset'
file = 'iris_scatter_matrix.pdf'
plt.title(title)
plt.savefig(file)
'''