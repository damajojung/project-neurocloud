import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import datasets

st.set_page_config(layout="wide")

st.title("Logistic Regression")

st.markdown(
    """
So far we have worked with regression models which predict a continuous output given inputs. With logistic regression, we work for the first time 
on a classification task which predicts a discrete ouput given inputs. The Classification tasks discussed on this website are Logistic regression, k-nearest
neighbours, decision threes and ranrom forests. Lets start with logistic regression and a dataset that needs no introduction in the ML world, namely the
iris dataset. For the people who never heard of this dataset, allow me to introduce it. All the other ones can skip the following passage. 
"""
)

st.header("Iris Dataset")

st.markdown(
    """
Three variations of Iris flowers are contained within the dataset with the following attributes: For each flower there is a petal and sepal length and width
which leads to a total of four attributes for each flower. The three different variations of Iris are Setosa, Versicolor and Virginica and are stored
in the dataset as 0,1 and 2. The dataset looks as follows:
"""
)

test = datasets.load_iris(as_frame=True)
test_df = test["data"]
test_df["Species"] = test["target"].astype(int)
st.dataframe(test_df)

st.markdown(
    """
With the data in mind, we can now proceed to the theory behind logistic regression. 
"""
)

st.header("Theory of Logistic Regression")

st.markdown(
    """
With logistic regression the probability that an instance belongs to a certain class is estimated. If the estimate is greater than 50%, then the model predicts that the 
instance belongs to that class which is referred to as *positive class* label with label 1. If it is predicted as not belonging to that class it is called *negative class* with 
label 0. Therefore, logistic regression is a binary classifier. This begs the question how logistic regression estimated probabilities? Same as in Linear Regression models, 
a Logistic Regression model computed the weighted sum of input features with a bias term. However, instead of outputting the results dierectly, the logistic of this result is
used. Therefore, it is bounded between 0 and 1 and is flexible to change the position or the steepness of thed curve which we will see in a short time.
"""
)

st.markdown(
    r"""
The probability that a certain instance belongs to class 1 can be written as follows:

$p(C_1, X) \approx \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$

The odds are:

$\frac{p(x)}{1 - p(X)} \approx e^{\beta_0 + \beta_1X}$

And the logg-odds (logit):

$log(\frac{p(X)}{1-p(X)}) \approx \beta_0 + \beta_1X$
"""
)

st.subheader("Estimate Coefficients: Maximum likelihood")

st.markdown(
    """
Let's go!
"""
)


# Functions
# Creating function for getting plots
def get_figure(x, y):

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.scatter(x, y, s=7)

    ax.set_xlabel("Petal length")
    ax.set_ylabel("Petal width")
    return fig, ax


# Data
@st.cache
def load_data():
    data = datasets.load_iris()
    return data


iris = load_data()

### Machine Learning

# Let’s try to build a classifier to detect the Iris virginica type based only on the petal width feature. First let’s load the data:

X = iris["data"][:, 3:]  # 3: petal width
y = (iris["target"] == 2).astype(int)  # 1 if Iris virginica, else 0

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X, y)

# Plot the result
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

fig, ax = get_figure(X, y)
plt.plot(X[y == 0], y[y == 0], "bs")  # blue square
plt.plot(X[y == 1], y[y == 1], "g^")  # green triangle
plt.plot([decision_boundary, decision_boundary], [-1, 2], "r:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
st.pyplot(fig)


##### Two Features - Softmax Regression

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(int)
log_reg = LogisticRegression(
    solver="lbfgs", C=10 ** 10, random_state=42
)  # The higher the level of C, the less the model is regularised
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1), np.linspace(0.8, 2.7, 200).reshape(-1, 1),
)
X_new = np.c_[
    x0.ravel(), x1.ravel()
]  # Creating a matrix with all the points which lie wihtin the two ranges

y_proba = log_reg.predict_proba(X_new)  # Get all the probabilities

fig, ax = get_figure(X[:, 0], X[:, 1])
plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")
plt.plot(X[y == 1, 0], X[y == 1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)  # Getting the contour lines


left_right = np.array([2.9, 7])  # boundries of x values
boundary = (
    -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
)

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
st.pyplot(fig)
