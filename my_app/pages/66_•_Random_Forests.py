import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sklearn
from sklearn import datasets
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn import tree

st.set_page_config(layout="wide")

st.title("Random Forests")

st.markdown(
    """
Random Forest is an ensemble of Decision Trees that are trained via the bagging method, typically with `max_samples`set to the size of the training set. 
Instead of training one tree perfetctly, we aim to train many trees imprefectly. Each of them trainied on a random subset of features and/or samples. 
Like with Decision Trees, you can perform classification as well as regression tasks by using `RandomForestClassifier`and `RandomForestRegressor`. 
Let's dive into the main ideas behind Random Forests.
"""
)

st.header("Theory")

st.markdown(
    r"""
Bagging, which stands for Bootstrap Aggregation, is a machine learning ensemble method that involves combining multiple models to improve the overall 
 accuracy and stability of predictions. In the context of random forests, bagging is used to create a diverse set of decision trees that can make robust
 and accurate predictions.

The theory behind bagging in random forests is based on two key principles:

* **Bootstrap Sampling**: The first principle of bagging is to use bootstrap sampling to create multiple datasets from the original data. Bootstrap sampling involves randomly sampling the original dataset with replacement, creating multiple datasets of the same size as the original but with some observations repeated and others left out. This process creates datasets that are slightly different from the original data, allowing for the generation of a diverse  set of decision trees.

* **Random Feature Selection**: The second principle of bagging in random forests is to randomly select a subset of features from the dataset for each tree.
  This helps to reduce the correlation between the trees and increases the diversity of the forest.

In random forests, multiple decision trees are trained on different bootstrap samples of the original data, each using a randomly selected subset of 
features. The final prediction is made by averaging the predictions of all the trees in the forest, giving equal weight to each tree. This process helps 
to reduce overfitting and increases the stability and accuracy of the model. Additionally, because each tree is trained on a slightly different dataset 
and uses a different subset of features, the model is less sensitive to outliers and noise in the data.
"""
)

st.header("AdaBoost")

st.markdown(
    r"""
AdaBoost, short for Adaptive Boosting, is a popular ensemble learning method that combines several weak models to form a strong model. In the context of 
random forests, AdaBoost can be used to improve the performance of the forest by boosting the importance of difficult-to-predict instances.

The theory behind AdaBoost in random forests is based on the following principles:

* Weighted Sampling: The first principle of AdaBoost is to use weighted sampling to adjust the importance of observations in the dataset. The weights are 
    initially set to 1/N, where N is the number of observations. After the first model is trained, the weights are adjusted to give more importance to 
    observations that were incorrectly classified and less importance to correctly classified observations.

* Combining Weak Models: The second principle of AdaBoost is to combine several weak models to form a strong model. Each weak model is trained on a 
    different weighted dataset, and the final model is formed by combining the predictions of all the weak models.

In random forests, AdaBoost can be used to improve the performance of the forest by boosting the importance of difficult-to-predict instances. 
The AdaBoost algorithm trains a series of decision trees, with each subsequent tree focusing on the instances that were incorrectly classified by the 
previous tree. The weights are adjusted after each tree, and the final prediction is made by combining the predictions of all the trees.

The benefit of AdaBoost is that it can improve the accuracy of the forest by focusing on difficult-to-predict instances. The weights of the observations 
ensure that the models focus on the observations that are difficult to classify correctly, improving the overall performance of the forest. Additionally, 
AdaBoost is less prone to overfitting than other ensemble methods, as it places more emphasis on difficult-to-predict observations, rather than relying 
too heavily on the most frequently occurring patterns in the data.

"""
)

st.header("XGBoost")

st.markdown(
    r"""
XGBoost, which stands for eXtreme Gradient Boosting, is a powerful ensemble learning method that combines multiple weak models to form a strong model. 
It is a variant of the gradient boosting method that uses a combination of additive tree models to improve the accuracy and speed of the forest.

The theory behind XGBoost in random forests is based on the following principles:

* Gradient Boosting: The first principle of XGBoost is gradient boosting. Gradient boosting involves the iterative addition of decision trees to the 
    ensemble, with each subsequent tree focusing on the errors made by the previous tree. The prediction of the final model is the sum of the predictions 
    of all the trees.

* Regularization: The second principle of XGBoost is regularization. Regularization is used to prevent overfitting by penalizing complex models. 
    Regularization can be achieved in XGBoost by adding a penalty term to the loss function.

* Feature Importance: The third principle of XGBoost is feature importance. Feature importance is a measure of how much each feature contributes to the 
    performance of the model. XGBoost can compute feature importance by calculating the number of times each feature is used in the trees.

* Parallel Processing: The fourth principle of XGBoost is parallel processing. XGBoost can parallelize the computation of the decision trees, 
    enabling it to train large ensembles quickly.

In XGBoost, the gradient boosting algorithm is extended by adding additional regularization terms to the loss function, which helps prevent overfitting. 
Additionally, XGBoost computes feature importance and can handle missing data, making it a robust and flexible method for solving a wide range of problems.

XGBoost is particularly useful for large-scale machine learning problems, as it can parallelize the computation of the decision trees and handle missing 
data effectively. Additionally, the feature importance measure allows users to identify the most important features in the dataset, enabling them to focus
 on the most relevant features for their problem. Overall, XGBoost is a powerful and flexible method for solving a wide range of machine learning problems.
"""
)

# Get data
dir = os.getcwd()
data_dir = dir + "/app_data/possum.csv"


@st.cache
def load_data(path):
    data = pd.read_csv(path)
    return data


df = load_data(data_dir)
df = df.dropna()

X = df.drop(["case", "site", "Pop", "earconch", "footlgth", "sex"], axis=1)
y = df["sex"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.header("Example")

st.markdown(
    r"""
Let's do it with real data. Here we have a possum data set which contains the following attributes `'case', 'site', 'Pop', 'sex', 'age', 'hdlngth', 'skullw', 'totlngth',
       'taill', 'footlgth', 'earconch', 'eye', 'chest', 'belly'` which describe certain characteristics of possums. The actual data can be seen in the 
       following data frame:
"""
)

st.dataframe(df)

st.markdown(
    r"""
Let us predict the sex of a possum with a Random Forest. This is done in the following section.
"""
)


st.header("Random Forest for Classification")

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=1000, max_features="auto", random_state=42
)
rf_model.fit(X_train, y_train)


predictions = rf_model.predict(X_test)
comparison = pd.DataFrame({"Predictions": predictions, "Actual": np.array(y_test)})
st.dataframe(comparison)

accuracy = accuracy_score(y_test, predictions)
st.write(accuracy)

# Export the first three decision trees from the forest

for i in range(3):
    tree_for = rf_model.estimators_[i]
    fig = plt.figure(figsize=(10, 7))
    _ = tree.plot_tree(
        tree_for, feature_names=X_train.columns, filled=True, rounded=True, max_depth=2
    )
    st.pyplot(fig)

cm = confusion_matrix(y_test, predictions, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax)

st.pyplot(fig)

importances = rf_model.feature_importances_
columns = X.columns
i = 0

while i < len(columns):
    st.caption(
        f"The importance of feature {columns[i]} is {round(importances[i] * 100,2)} %."
    )
    i += 1


####### Cross Validation

param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
st.write(rand_search.best_params_)

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary", pos_label="m")
recall = recall_score(y_test, y_pred, average="binary", pos_label="m")

st.write("Accuracy:", accuracy)
st.write("Precision:", precision)
st.write("Recall:", recall)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(
    best_rf.feature_importances_, index=X_train.columns
).sort_values(ascending=False)

# Plot a simple bar chart
fig = plt.figure(figsize=(15, 10))
feature_importances.plot.bar()
st.pyplot(fig)
