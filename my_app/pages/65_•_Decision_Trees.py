import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sklearn
from sklearn import datasets
import os
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")

st.title("Decision Trees")

st.markdown(
    """
Decision Trees.
"""
)

from sklearn.datasets import load_wine

data = load_wine(as_frame=True)

X = data.frame.loc[:, ["alcohol", "color_intensity"]]
y = data.frame.iloc[:, -1:]

tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X, y)

from sklearn import tree

fig = plt.figure(figsize=(15, 10))
_ = tree.plot_tree(
    tree_clf,
    feature_names=["alcohol", "color_intensity"],  # wine.feature_names
    class_names=["class_0", "class_1", "class_2"],  # wine.target_names
    filled=True,
)
st.pyplot(fig)

# Useful information:
# Normally, the data must be numerical. However, one can onehotencode the data as follows:

# one_hot_data = pd.get_dummies(dat[['Outlook', 'Temperature', 'Humidity', 'Windy']],drop_first=True)
# tree_clf.fit(one_hot_data, dat['Play'])

# And then you can use it as well. I learned also something interesting. Nested list with 3 elements

# targets = [{'no': 0, 'yes': 1}.get(i, 'none') for i in list(dat['Play'])]
# dat["Play"] = targets
