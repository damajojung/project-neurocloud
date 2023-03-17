import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import seaborn as sns
import sklearn
from sklearn import datasets
import os

plt.style.use(["default"])

mpl.rc("axes", labelsize=8)  # Labels of the axes
mpl.rc("xtick", labelsize=8)  # x-ticks
mpl.rc("ytick", labelsize=8)  # y-ticks

st.set_page_config(layout="wide")

st.title("PCA")

st.header("The curse of Dimensionality")

st.markdown(
    r"""
In a world of big data, one is pretty quickyl confronted with datasets that contain thousand if now even millions of features for earch training instance.
Thus, training can become really slow and finding a good solution hard. This problem is called *The curse of dimensionality*. Moreover, apart from 
reducing running times during training, dimensionality reduction is also very beneficial for data vizualisation. Reducing the dimensions down to
two or three dimensions allows one to plot the data and potentially detect any clusters or patterns within the data. DataViz is crucial for communicating
your conclusions to people who do not have a background in data science. We will take a look at one of the most common dimensionality reduction techniques
within data science, namely *Principle Component Analysis*. 

"""
)

st.header("Two Approaches for Dimensionality Reduction")

st.markdown(
    r"""
There are currently two main approaches when it comes to dimensionality reduction, namely **Projection** and **Manifold Learning**. The former one takes for example 3D data and 
projects it down to two dimensions. Imagine a parabola in 3D which can easily be projected down to a plane. In certain cases this approach works just fine. However,
you can imagine that not all data sets are as simple as a parabola. If the data overlaps along the axis one likes to project the data you loose a lot of information. Therefore, 
manifold learning is more bulletproof. Many dimensionality reductions algorithms work by modeling the manifold on which the training instances lie which is calles *Manifold
Learning*. The underlying assumption is calles *manifold hypothesis* which holds that most real-world-high-dimensional datasets lie close to a much lower-dimensional manifold. 
*Principal Component Analysis (PCA)* is by far the most populat dimensionality reduction algorithm and is discussed in the following section.
"""
)

st.markdown(
    r"""
PCA first identifies the hyperplane that lies closest to the data and then it projects the data onto it. In order to preserve as much as possible from the variance
of the data one has to find the right hyperplane. This can be done by minimizing the mean squared distance between the original dataset and its projection onto the axis
it is projected on. This is basically the main concept behind PCA. We will not shed light onto the mathemtatics behind PCA here since there are plenty of helpful 
recourses for that. However, we will take a look at an example. 
"""
)

st.header("Dimensionality Reduction for Wine Data")

st.markdown(
    r"""
We use another very commong data set within the data science world which is the wine dataset. It consists of 13 variables and 178 samples and contains 3 classes of wine.
I certainly do like to drink wine, but those chemicals within the data do not say that much to me. However, as mentioned before, It can be very helpful to reduce
the dimensionality of the data in order to make some DataViz and to identify any clusters. Let's go through it step by step. 
"""
)

# Get the data
from sklearn.datasets import load_wine

data = load_wine(as_frame=True)

X = data.frame.loc[:, :]
y = data.frame.iloc[:, -1:]

st.subheader("1.) Take a look at the data")

st.markdown(
    r"""
It's always advisable to take a look at the data first. Besides the target variable, all the other ones are of numerical nature. This is important
since PCA only works with numerical data. However, we can see that the data is not yet centered around the origin and scaled to unit variance. This is important. Even though
Scikit learn does center the data and scale it automatically for us, we'll do it by hand this time in order to see each step."""
)

# Display the data
st.dataframe(X)

# Standardise everythinbg - Mean 0, standard deviation 1
from sklearn.preprocessing import StandardScaler

st.subheader("2.) Center and scale the Data")

st.markdown(
    r"""
Centering the data around the origin plus scaling can easily be done with `StandardScaler().fit_transform(X)` and the result looks as follows:"""
)

# Standardizing the features
x = StandardScaler().fit_transform(X)
st.dataframe(x)

st.markdown(
    r"Please note that the target variable has been excluded from this procedure."
)

# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])

# Merge
final_df = pd.concat([principalDf, y], axis=1)

st.subheader("3.) Conducting PCA")

st.markdown(
    r"""
Now, all we have to to is to specify to how many dimensions we would like to reduce the data set to by using `PCA(n_components=2)`. And then, we can feed the
data into it with `pca.fit_transform(x)`. Lastly, one has to add the target variable to the PCA variables which looks as follows:"""
)

st.dataframe(final_df)

st.subheader("4.) Plot the results")

st.markdown(
    r"""
Finally, we can plot the data and color it by their target value. We can see that PCA was able to find a solutions that allows us to visualize a 13 dimensional data set
onto a two dimensional plane. Now, one can conduct further analysis if he or she wishes to do so.
"""
)

# Display everything
fig, ax = plt.subplots(1, figsize=(15, 10))
plt.plot(
    final_df[final_df.target == 0]["PC1"],
    final_df[final_df.target == 0]["PC2"],
    "ro",
    label="Wine Class 1",
    markersize=10,
)
plt.plot(
    final_df[final_df.target == 1]["PC1"],
    final_df[final_df.target == 1]["PC2"],
    "bs",
    label="Wine Class 2",
    markersize=10,
)
plt.plot(
    final_df[final_df.target == 2]["PC1"],
    final_df[final_df.target == 2]["PC2"],
    "g^",
    label="Wine Class 3",
    markersize=10,
)
plt.xlabel("Principal Component 1", fontsize=20)
plt.ylabel("Principal Component 2", fontsize=20)
plt.legend(loc="upper left", fontsize=20)
st.pyplot(fig)

