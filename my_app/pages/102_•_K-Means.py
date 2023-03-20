import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sklearn
from sklearn import datasets
import os
from sklearn.datasets import make_blobs

st.set_page_config(layout="wide")

#### Funtions


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], "k.", markersize=2)


def plot_centroids(centroids, weights=None, circle_color="w", cross_color="k"):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        s=35,
        linewidths=8,
        color=circle_color,
        zorder=10,
        alpha=0.9,
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=2,
        linewidths=12,
        color=cross_color,
        zorder=11,
        alpha=1,
    )


def plot_decision_boundaries(
    clusterer,
    X,
    resolution=1000,
    show_centroids=True,
    show_xlabels=True,
    show_ylabels=True,
):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution),
        np.linspace(mins[1], maxs[1], resolution),
    )
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel1")
    plt.contour(
        Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors="k"
    )
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


#### End Functions

st.title("K-Means")

st.markdown(
    r"""
K-means clustering is a widely used unsupervised machine learning algorithm that aims to partition a given dataset into K distinct, 
non-overlapping clusters based on the similarity of the data points. The algorithm works by iteratively assigning each data point to one of the 
K clusters and then computing the centroid of each cluster based on the mean of all the points assigned to that cluster. The centroids then become 
the new centers of the clusters, and the process is repeated until the centroids no longer move or a maximum number of iterations is reached.

The k-means algorithm is commonly used in various applications, including image segmentation, market segmentation, customer segmentation, and anomaly
 detection. It is a fast and efficient algorithm that can handle large datasets and is relatively easy to implement. However, it also has some 
 limitations, such as its sensitivity to the initial choice of centroids and its tendency to converge to a local minimum rather than the global 
 minimum. Let's explore k-Means with an example. 
"""
)

st.header("Theory")

st.markdown(
    r"""

The theory behind k-means is fairly simple. 

* Choose the number of clusters (k) you want to create.
* Randomly assign each data point to a cluster.
* Calculate the centroid (mean) of each cluster.
* Reassign each data point to the cluster whose centroid is closest.
* Repeat steps 3-4 until the clusters no longer change or a maximum number of iterations is reached.
* Optionally, use various metrics to evaluate the quality of the resulting clustering.

The commonly performance measurement looks as follows:

$\sum^{n}_{i = 1} (x_i - z_i)^2 = minimal$

where $z_i$ is the centroid that is closest to the sample $i$. Important: The solution for the performance measurement is not convex! Meaning:
The solution has some randomness in it and is dependent on the initialisation. We'll get back to that later. Let us explore it with an example.
"""
)

st.header("Example")

st.markdown(
    r"""
Here we can see an unlabeled data set composed of five blobs of data. With the help of k-means it should be pretty easy to find five clusters. Let's run sklearn's
k-means algorhitm on the data by using `KMeans(n_clusters=5, random_state=42)` and take a look at the result. Please note that $k$ must be specified beforehand and
if one wishes to optain reproducable results, one should also specify a seed.
"""
)
blob_centers = np.array(
    [[-0.5, -0.5], [-1.5, 2.5], [-3.0, 0.5], [-3.3, 2.8], [-2.7, -1.0]]
)
blob_std = np.array([0.3, 0.3, 0.15, 0.3, 0.2])

X, y = make_blobs(
    n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7
)


fig = plt.figure(figsize=(8, 4))
plot_clusters(X)
st.pyplot(fig)

######

from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

# st.caption(y_pred)
# st.caption(y_pred is kmeans.labels_) # True

# The following 5 centroid were calculated:

st.markdown(r"""We get the following five centroids:""")
kmeans.cluster_centers_

st.markdown(r"""with the following predictions for every sample:""")
kmeans.labels_

# Predict the labels for new instances
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)


st.markdown(r"""Which results in the following solution:""")
fig = plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
st.pyplot(fig)


###

st.title("Finding the right k (amount of clusters)")

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

fig = plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate(
    "Elbow",
    xy=(5, inertias[4]),
    xytext=(0.55, 0.55),
    textcoords="figure fraction",
    fontsize=16,
    arrowprops=dict(facecolor="black", shrink=0.1),
)
plt.axis([1, 8.5, 0, 1300])
st.pyplot(fig)


# Using Clustering for Image Segmentation could be something that could be interesting to look at

