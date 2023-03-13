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

st.header("Two approaches of dimensionality reduction")

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
PCA - Principal Component Analysis. Data must be standardised with mean 0 and standard deviation of 1.
And only numerical data works. 
"""
)

# Get the data
from sklearn.datasets import load_wine

data = load_wine(as_frame=True)

X = data.frame.loc[:, :]
y = data.frame.iloc[:, -1:]

# Display the data
st.dataframe(X)

# Standardise everythinbg - Mean 0, standard deviation 1
from sklearn.preprocessing import StandardScaler

# Standardizing the features
x = StandardScaler().fit_transform(X)
st.dataframe(x)

# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])

# Merge
final_df = pd.concat([principalDf, y], axis=1)

st.dataframe(final_df)

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

