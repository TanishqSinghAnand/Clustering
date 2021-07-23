from random import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
import math
import plotly.express as px


df = pd.read_csv("data.csv")
mass = df["mass"].tolist()
radius = df["radius"].tolist()

print(df.head())
print(mass[1])
print(radius[1])


X = []
for index, planetMass in enumerate(mass):
    if math.isnan(planetMass) == True:
        planetMass = 0
        mass[index] = 0
    if math.isnan(radius[index]) == True:
        radius[index] = 0
    tempList = [
      radius[index],
      planetMass
    ]
    X.append(tempList)

wcss = []
for i in range(1, 11):
  kMeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kMeans.fit(X)
  wcss.append(kMeans.inertia_)
plt.figure(figsize=(10, 5))
sns.lineplot(range(1, 11), wcss, markers='o', color='green')
plt.title("Elbow Method")
plt.xlabel('Number of clusters')
plt.ylabel('Wcss')
plt.show()


fig = px.scatter(x=radius, y=mass, color=mass)
fig.show()
