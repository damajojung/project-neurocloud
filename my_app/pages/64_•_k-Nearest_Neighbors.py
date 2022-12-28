import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn import datasets

st.set_page_config(layout="wide")

st.title("k-Nearest Neighbours")

st.markdown(
    """
k-Nearest Neighbours.
"""
)

# Creating function for getting plots
def get_figure(x, y, color, light, bold):

    fig, ax = plt.subplots(1, figsize=(10, 6))
    # m = [{0: "^", 1: "."}.get(i, "+") for i in y] # Very cool way to create a nested list with 3 elements
    plt.pcolormesh(xx, yy, Z, cmap=light)
    ax.scatter(x, y, s=30, c=color, cmap=bold, edgecolor="black")

    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")
    ax.legend()
    return fig, ax


# Getting data
@st.cache
def load_data():
    data = datasets.load_iris()
    return data


iris = load_data()

X = iris.data[:, :2]  # sepal length, sepal width
y = iris.target

h = 0.02

# Create color maps
cmap_light = ListedColormap(["#ffeac4", "#d8ffd8", "#dbeeff"])
cmap_bold = ListedColormap(["#ffb327", "#009d00", "#005db4"])

# calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

############## 1
n_neighbours = 1

# we create an instance of Neighbours Classifier and fit the data.
clf = KNeighborsClassifier(n_neighbors=n_neighbours)
clf.fit(X, y)

# predict class using data and kNN classifier
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig, ax = get_figure(X[:, 0], X[:, 1], color=y, light=cmap_light, bold=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"3-Class classification (k = {n_neighbours})")
st.pyplot(fig)

############## 5
n_neighbours = 5

# we create an instance of Neighbours Classifier and fit the data.
clf = KNeighborsClassifier(n_neighbors=n_neighbours)
clf.fit(X, y)

# predict class using data and kNN classifier
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig, ax = get_figure(X[:, 0], X[:, 1], color=y, light=cmap_light, bold=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"3-Class classification (k = {n_neighbours})")
st.pyplot(fig)

############## 15
n_neighbours = 15

# we create an instance of Neighbours Classifier and fit the data.
clf = KNeighborsClassifier(n_neighbors=n_neighbours)
clf.fit(X, y)

# predict class using data and kNN classifier
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig, ax = get_figure(X[:, 0], X[:, 1], color=y, light=cmap_light, bold=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"3-Class classification (k = {n_neighbours})")
st.pyplot(fig)
