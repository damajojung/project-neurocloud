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
    r"""
Decision Trees are powerful Machine learning algorithms that can perform both classification and regression tasks which are non-parametric and non-linear. 
The main advantages are that they are easy to visualize which is useful for interpretation. This is called a white box model. Moreover, they are fast
to train and evaluate which is especially beneficial when you are dealing with huge data sets. Additionaly, since they are non-parametric, no
pre-assumptions about the data is needed. However, they do also have a disadvantage. They have a strong tendency to overfit the training data
which leads to poor generalisation. Nonetheless, they are a fundamental component of Ranodm Forests which is among the most powerful ML algorithms
today. Therefore, it worthwhile taking a closer look at decision trees. Let's have a look at the theory. 
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
