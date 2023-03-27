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


def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    # plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


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

The commonly performance measurement looks as follows and is called inertia:

$\sum^{n}_{i = 1} (x_i - z_i)^2 = minimal$

where $z_i$ is the centroid that is closest to the sample $i$. Important: The solution for the performance measurement is not convex! Meaning:
The algorithm is guanranteed to converge, it is not guaranteed that it will converge to the right solution (i.e. it might converge to a local optimum). This all depends 
on the random initialisation of the centroids in the beginning. We'll get back to that later. Let us explore it with an example.
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
blob_std = np.array([0.3, 0.4, 0.15, 0.15, 0.2])

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

st.markdown(r"""After running the code we get the following five centroids:""")
kmeans.cluster_centers_

st.markdown(r"""with the following clusters for every sample:""")
kmeans.labels_

# Predict the labels for new instances
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)


st.markdown(r"""Which results in the following solution:""")
fig = plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
st.pyplot(fig)

st.markdown(
    r"""

Please note that the big X in the middle of the blogs represent the centroids. Overall, the results look pretty good. Only a few sample were probably mislabeled.
Take a closer look at the boundary between both top clusters. This is a common problem when the k-means algorithm is faced with blobs of different diameters of blobs since
it only cares about the distance to the centroids. Well, there are now two questions in the room that must be adressed. How does one find the right $k$ if it can not be
clearly seen from a plot? And what about the randomnees of the centroids initalisation? 
"""
)

### Additional Exploration

st.title("Finding the right k (amount of clusters)")

st.markdown(
    r"""

We have seen that getting k-means up and running with an obvious example is straight forward. But if we have more dimensions where it is not possible to visualise it we
need a different approach for selecting a good value for k because the algorithm alsways finds a solution for us which can be seen in the following figure where we use
the same data but use $k = 3$ and $k = 8$. 
"""
)

kmeans_k3 = KMeans(n_clusters=3, random_state=42)
kmeans_k8 = KMeans(n_clusters=8, random_state=42)

fig, ax = plt.subplots(figsize=(8, 3.5))
plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")
st.pyplot(fig)

### Elbow plot

st.markdown(
    r"""
With the help of the computed inertia it is possible to find an optimal k. But answer is not as straight forwards as it might appear. You would think that the lowest ineria
score is the best, right? Yeah, this is kind of true, but the problem is that the inertia score decreases with bigger k. Therefore, one should choose a score wich decreases
the inertia sccore enough to an acceptable level but does not overfit. We can plot the scores for different k's which can be seen in the following figure. A good k is
where you can see a knick in the plot which is called the elbow since it resembles to an arm. Any lower k would be bad since inertia could be decreased a lot simply by
going one or two higher and any higher value would not help that much either. However, the knick must be clearly visible in order to find a good k. 
"""
)

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


st.title("Centroid initialisation")

st.markdown(
    r"""

Another crutial part about k-means is the centroid initialisation. Normally, the centroids are selected randomly at the beginning. However, you can see in the following
figure that the result can be very different with a different initialisation. 
"""
)

kmeans_rnd_init1 = KMeans(
    n_clusters=5, init="random", n_init=1, algorithm="full", random_state=2
)
kmeans_rnd_init2 = KMeans(
    n_clusters=5, init="random", n_init=1, algorithm="full", random_state=5
)

fig, ax = plt.subplots(figsize=(8, 3.5))
plot_clusterer_comparison(
    kmeans_rnd_init1,
    kmeans_rnd_init2,
    X,
    "Solution 1",
    "Solution 2 (with a different random init)",
)
st.pyplot(fig)

st.markdown(
    r"""
So how can we solve this? If you happen to know approximately where the centroids should be located, then you can set the init hyperparameter to a NumPy array 
containing the list of centroids, and set n_init to 1. This ensures that the algorithm starts roughly at the right position. Another solution is to run the algorithm
multiple times and keep the best solution by analysing the inertia value. The number of random initializations is controlled by the n_init hyperparameter: by default, 
it is equal to 10, which means that the whole algorithm described earlier runs 10 times when you call fit(), and Scikit- Learn keeps the best solution.
"""
)

st.title("""K-means and Images""")


from PIL import Image

image = Image.open("app_data/k_means_tulip.png")

st.markdown(
    r"""
Image segmentation is the process of dividing an image into multiple regions or segments, each of which represents a different object or region of interest within the image.
The goal of image segmentation is to simplify or change the representation of an image into something that is easier and more meaningful to analyze.
In other words, image segmentation is a technique used in computer vision and image processing to identify and extract objects or regions of interest from an image. It 
can be used for a variety of applications, such as object recognition, image editing, medical imaging, and autonomous driving.

There are various approaches to image segmentation, including thresholding, edge detection, region growing, and clustering. Each approach has its own advantages and 
disadvantages and can be selected based on the specific requirements of the application. However, we are interested in color segmentation. Color 
segmentation is a specific type of image segmentation that involves segmenting an image based on its color 
information. In color segmentation, an image is divided into multiple segments or regions based on the colors present in the image. Color 
segmentation is a popular approach for image segmentation in computer vision and image processing applications, as color is a prominent visual 
feature that can be used to distinguish between different objects or regions of interest within an image.

The following figure shows k-means with different $k$ on the same image. We can see that orange tulip in the background is replaced with green and violet going from $k=6$
to $k=4$ since it is a smaller cluster compared with the other ones. 

"""
)

st.image(image, caption="K-means on tulip")

