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

A smaller k leads to fewer misclasification within the training set. $k=1$ leads to no misclasification. However, this is heavily overfitting the training data set. Therefore,
one should always evaluate the performance with a test data set. This leads to the assumption that choosing a too small or too big $k$ is unfavorable and one has to find
the right one somewhere in between. 
"""
)

st.subheader("Practical Information")

st.markdown(
    r"""
* If you have two classes, always choose an uneven $k$ in order to avoid ambiguous solutions
* With more than two classes: it is not always possible to obtain a clear clisification. If it ends in a draw, the claisification is randomly chosen
* It is advisable to use a seed in order to obtain reproducable results
* The **advantage** of this method lies in its simplicity
* The **disadvantages** of the method are that one needs alwas the whole training set. Moreover, the higher the dimensionality of the samples the more samples you need in order to 
    obtain clear results
* The distance measurement is the Euclidean metric and important: This method only works with metric data 
"""
)

st.header("k-Nearest-Flowers")

st.markdown(
    r"""
Since we have worked with the Iris dateset in the previous section, let me illustrate the impact of different $k$s, namely 1,5 and 15 on the Iris data set with features
sepal width and length. 
"""
)

# Creating function for getting plots
def get_figure(x, y, color, light, bold):

    fig, ax = plt.subplots(1, figsize=(10, 6))
    # m = [{0: "^", 1: "."}.get(i, "+") for i in y] # Very cool way to create a nested list with 3 elements
    plt.pcolormesh(xx, yy, Z, cmap=light)
    ax.scatter(x, y, s=30, c=color, cmap=bold, edgecolor="black")

    ax.set_xlabel("Sepal length", fontsize=14)
    ax.set_ylabel("Sepal width", fontsize=14)
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

st.subheader("Performance Evaluation")

st.markdown(
    r"""
It is clearly visible that by choosing a $k=1$, there are still quite a few islands and the borders are wiggly. These are flattened out with $k=5$ and barely occur anymore with
$k=15$. This begs the question: What is the optimal value for k?

This is a classic optimization problem which can be solved with the values of the confusion matrix. We need a performance measurement which can be minimized or maximized, 
depending on the measurement. There are many performance measurements. However, these are some of the most commonly used.  

| Metric | Formula |
|--------|---------|
| Sensitivity (True Positive Rate) | (True Positives) / (True Positives + False Negatives) |
| Specificity (True Negative Rate) | (True Negatives) / (True Negatives + False Positives) |
| Accuracy | (True Positives + True Negatives) / (Total number of instances) |
| Error Rate (Misclassification Rate) | (False Positives + False Negatives) / (Total number of instances) |

We will be working with the error rate which can also be written as $1 \, - \, accuracy$. There are three ways to use the error rate in order to find an optimal k:

* 1.) Use the confusion matrix to calculate the **in-sample error rate**. This method tends to underestimate the error rate. Therefore, it is not advisable to use this one. 
* 2.) Calculate the **out-of-sample error rate** by using a confusion matrix created by a test data set. This procedure provides an unbiased estimate. However, one needs
    quite a lot of samples in order to achieve this. 
* 3.) The state of the art is to use **cross validation** where the whole dataset is split randomly into a train and test data set and with the test data sets the error rate 
    is calculated. A special case is the **jackknife** or **leave-one-out** method where the sample only contains one object. The validation is completed once every object has
    been left out. The only drawback of this method is that it is rather computationally intensive and the variance is rather large in smaller data sets. Therefore, it is
    advisable to leave several objects out. 
"""
)

st.subheader("Leave-one-out cross validation")

st.markdown(
    r"""
Let me illustrate the leave-one-out cross validation with our data set where for each iteration the negative mean absolute error is calculated and plotted for the 
different $k$ which can be seen in the following figure. On the x-axis are the k values ranging from 0 to 40 and on the y-axis are the negative means absolute errors. 
The goal is to take a k with a performance score as low as possible. Ideally, k should also be as small as possible. this is the case for $k = 30$. The result of of $k = 30$
can be found on the final figure.
"""
)

### Getting the best k
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

error_rate = []
# Will take some time
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    cv = LeaveOneOut()
    score = cross_val_score(knn, X, y, cv=cv, scoring="neg_mean_absolute_error")
    error_rate.append(np.mean(np.absolute(score)))

fig1 = plt.figure(figsize=(10, 6))
plt.plot(
    range(1, 40),
    error_rate,
    color="black",
    linestyle="dashed",
    marker="o",
    markerfacecolor="white",
    markersize=8,
)

plt.ylabel("Leave-one-out cross validation score", fontsize=14)
plt.xlabel("k", fontsize=14)

st.pyplot(fig1)

# And now the final figure

############## 30
n_neighbours = 30

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

# I could still add Sensitivity, Specificity, Matthews correlation coefficient, Cramers V and the Kappa statistic. Maybe in the future.

