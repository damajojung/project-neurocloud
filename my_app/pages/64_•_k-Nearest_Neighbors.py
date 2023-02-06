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
    r"""
The foundation of k-nearest neighbours is the *nearest-neighbour* method. This is a simple classification method that takes a training data set which contains a sample vector
 $x_i$ with a class index $k_i$ for each sample $i = 1,...,n$. Each new  sample $x_s$ with unknown class index is assigned to the same class as the k nearest neighbours. 
 The nearest neighbour $x_{j_{s}}$ of $x_s$ can be written as follows:

$j_s = arg \, min_{i=1,...,n} \, d(x_s, x_i)$

Where d is a measurement of distance, most often the Euclidean distance metric.

Logically, an extension to nearest-neighbour method is the k-nearest neighbour (kNN) method where not only the nearest neighbour is taken into consideration but the k-nearest
neighbours of sample $x_o$. Among the k-nearest-neighbours of $x_o$, let each $m_l$ of class $l , l = 1,...,g$. So $\sum_{l=1}^{g} m_l = k$ applies. The observation 
$x_o$ is now assigned to the class $\hat{k} (\in 1, ..., g)$ for which holds:

$m_{\hat{k}} = max_{l=1,...,g} \, m_l$

The bigger $k$...
* the clearer one can see areas of the same class
* the smaller is the variability within the estimate of the border of the classes
* the bigger is the systematic error (bias of classification)
"""
)

st.header("Misclassification")

st.markdown(
    r"""
Bias always leads to misclassification. Misclassification measures the amount of misclassification and is calculated by taking the amount of misclassified samples divided by the
 total classified samples which can be represented by a confusion matrix. 

 |               | Actual Positive | Actual Negative |
|:-------------:|:--------------:|:--------------:|
| Predicted Positive  | TP (True Positive) | FP (False Positive) |
| Predicted Negative  | FN (False Negative) | TN (True Negative) |

Here, TP refers to the number of true positive predictions, FP refers to false positive predictions, FN refers to false negative predictions, and TN refers to
 true negative predictions. So for example, if we take $k=1$, we could get a matrix like the following one:

 |               | Actual Positive | Actual Negative |
|:-------------:|:--------------:|:--------------:|
| Predicted Positive  | 100            | 0              |
| Predicted Negative  | 0              | 100            |

where we have an error of 0. However, if we increase k to $k=10$, then it results in the following confusion matrix:

|               | Actual Positive | Actual Negative |
|:-------------:|:--------------:|:--------------:|
| Predicted Positive  | 90             | 10             |
| Predicted Negative  | 7              | 93             |

Here, 90 out of 100 positive instances have been correctly predicted as positive, 10 have been predicted as negative (false negatives). Out of 100 negative instances, 
93 have been correctly predicted as negative and 7 have been predicted as positive (false positives). The total error rate is calculated as (10 + 7)/(100 + 100) = **0.067**.


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
