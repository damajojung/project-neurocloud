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

st.markdown(
    """
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

