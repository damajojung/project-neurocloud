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

Let me illustrate the impact of different $\beta_0$ and $\beta_1$.
"""
)

########################## Illustrate the impacts of beta_0 and beta_1

##### 1

x = np.linspace(-5, 5, 50)
beta_0 = 0
beta_1 = 1
term = np.exp(beta_0 + beta_1 * x)

sig = term / (1 + term)
fig = plt.figure(figsize=(9, 3))
plt.plot([-5, 5], [0, 0], "k-")
plt.plot([-5, 5], [0.5, 0.5], "k:")
plt.plot([-5, 5], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(
    x,
    sig,
    "b-",
    linewidth=2,
    label=r"$p(C_1, X) \approx \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$",
)
plt.xlabel("X")
plt.legend(loc="upper left", fontsize=17)
plt.axis([-5, 5, -0.1, 1.1])

beta_0 = -1
beta_1 = 1
term = np.exp(beta_0 + beta_1 * x)
sig = term / (1 + term)
plt.plot(x, sig, "r-", linewidth=2)

plt.text(2.25, 0.37, r"$\beta_0$", fontsize=14, color="r", ha="center")
plt.arrow(2, 0.4, -0.5, 0, head_width=0.05, head_length=0.1, fc="r", ec="r")
plt.arrow(2.5, 0.4, 0.5, 0, head_width=0.05, head_length=0.1, fc="r", ec="r")

st.pyplot(fig)


#### 2

x = np.linspace(-5, 5, 50)
beta_0 = 0
beta_1 = 1
term = np.exp(beta_0 + beta_1 * x)

sig = term / (1 + term)
fig = plt.figure(figsize=(9, 3))
plt.plot([-5, 5], [0, 0], "k-")
plt.plot([-5, 5], [0.5, 0.5], "k:")
plt.plot([-5, 5], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(
    x,
    sig,
    "b-",
    linewidth=2,
    label=r"$p(C_1, X) \approx \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$",
)
plt.xlabel("X")
plt.legend(loc="upper left", fontsize=17)
plt.axis([-5, 5, -0.1, 1.1])

beta_0 = 0
beta_1 = 4.0
term = np.exp(beta_0 + beta_1 * x)
sig = term / (1 + term)
plt.plot(x, sig, "g-", linewidth=2)

plt.text(2, 0.39, r"$\beta_1$", fontsize=14, color="g", ha="center")
plt.arrow(2, 0.53, 0, 0.1, head_width=0.15, head_length=0.1, fc="g", ec="g")
plt.arrow(2, 0.32, 0, -0.1, head_width=0.15, head_length=0.1, fc="g", ec="g")

plt.arrow(2.3, 0.53, 0.5, 0.15, head_width=0.1, head_length=0.1, fc="b", ec="b")
plt.arrow(1.7, 0.32, -0.5, -0.15, head_width=0.1, head_length=0.1, fc="b", ec="b")

st.pyplot(fig)


### 3

x = np.linspace(-5, 5, 50)
beta_0 = 0
beta_1 = 1
term = np.exp(beta_0 + beta_1 * x)

sig = term / (1 + term)
fig = plt.figure(figsize=(9, 3))
plt.plot([-5, 5], [0, 0], "k-")
plt.plot([-5, 5], [0.5, 0.5], "k:")
plt.plot([-5, 5], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(
    x,
    sig,
    "b-",
    linewidth=2,
    label=r"$p(C_1, X) \approx \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$",
)
plt.xlabel("X")
plt.legend(loc="upper left", fontsize=17)
plt.axis([-5, 5, -0.1, 1.1])

beta_0 = 0
beta_1 = 0.05
term = np.exp(beta_0 + beta_1 * x)
sig = term / (1 + term)
plt.plot(x, sig, "y-", linewidth=2)

plt.text(2.25, 0.37, r"$low \, \beta_0$", fontsize=14, color="y", ha="center")
plt.arrow(1.7, 0.4, -0.5, 0, head_width=0.05, head_length=0.1, fc="y", ec="y")
plt.arrow(2.8, 0.4, 0.5, 0, head_width=0.05, head_length=0.1, fc="y", ec="y")

st.pyplot(fig)

st.subheader("Estimate Coefficients: Maximum likelihood")

st.markdown(
    r"""
With Linear Regression there is a closed-form solution for the optimiztion problem since least squares results in minimizing a quadratic function. However, a closed-form
solution is hard to obtain with a logistic function. The goal is to find a model that maximizes the probability of observing the data at hand. This is being done with 
maximum likelihood with the following likelihood function:

$\ell(\beta_0, \beta_1) = \prod_{i \in C_1}^{} p(C_1 | x_i) \prod_{i \in C_2}^{} (1-p(C_1 | x_i))$

This likelihood function describes the probability that data $X_1 \in C_1$ and $X_2 \in C_2$ is observed given the parameters $\beta_0, \beta_1$. Now, one has to
compute $\beta_0, \beta_1$ that maximize $\ell(\beta_0, \beta_1)$ which equivalent to minimizing the negative log-likelihood (error function, cross-entropy error):

$\mathcal{L}(\beta_0, \beta_1) = - \sum_{i \in C_1}^{} log(p(X_i)) - \sum_{i \in C_2}^{} log(1 - p(x_i))$

**Goal:** Compute $(\beta_0, \beta_1)$ that minimize the log-likelihood. This can be achieved by using the numerical optimization method *gradient descent*. To give a 
little more mathematical context, a gradient is a vector of partial derivatives which points into the direction of the steepest ascent. On the other hand, the negative
gradient point into the direction of the steepest descent. How convenient, we are trying to find the minimum of a cost function!  

**How to apply gradient descent?**
1. Start with initial guess of parameters $\bm{\beta}$
2. Uptdate according to: $\bm{\beta}_{t+1} = \bm{\beta}_t - \gamma (\nabla f (\bm{\beta}_t))$

Where $\gamma$ is the step size and $f$ is the loss/cost function (i.e., MSE or negative log-likelihood) and: 

$\nabla f (\bm{\beta}_t) = \sum_{i}^{N} (p(x_i) - y_i)\bm{x_i}$

$p(x_i) = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$
"""
)

st.subheader("Stochastic gradient descent (SGD)")

st.markdown(
    r"""
The optimization process described in the previous sections has one flaw, namely that all the data is required to compute the gradient (N) which can turn out to be
very time consuming. Therefore, a solution to this obstacle is to randomly choose a subset of the data to estimate the gradient which is called a **mini-batch**. 
This is being done in stocahtic gradient descent where a randomely subset of the training data (mini-batch) is used to estimate the gradient. The solution still 
converges and the sampled gradient is an unbiased estimate of the true gradient. Phew, that was a lot of theory! Let's implement it! 
"""
)

st.subheader("Iris Virginia Classifier")

st.markdown(
    r"""
In this little experiment, I want to illustrate how one can use logistic regression to build a classifier that detects the *Iris virginia* type based only on the petal width 
feature. The reange of interest of the the feature petal width lies within 0 cm and 3 cm. The estimated probabilities of the model look as follows:
"""
)

# Functions
# Creating function for getting plots
def get_figure(x, y):

    fig, ax = plt.subplots(1, figsize=(10, 4))
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

st.markdown(
    r"""
We can see that the petal width of *Iris virginica* flowers range from 1.4 cm to 2.5 cm and are represented by the green triangles in this plot. On the other hand, other iris 
flowers (represented by the blue squares) tend to have smaller petal width that range between 0.1 cm and 1.8 cm. Our classifier is pretty confident that a flower with a petal width
above 2 cm is of type *Iris virginica* since it outputs a high probability for that class. Under 1 cm it is pretty confident that the flower is **not** of type *Iris virginica*
since the probability for 'Not Iris virginica' has a high probability. Please note that that there is a bit of overlap in the middle which results in uncertainty.  The 
decision boundary is at around 1.6 cm where the probability is 50%. If the petal width is above 1.6 cm, the classifier classifies a sample as *Iris virginica*. On the other
side, if its smaller, then it's classified as 'Not Iris virginica'. 
"""
)

st.subheader("Multiclass Logistic Regression")

st.markdown(
    r"""
One can also use more then one feature for the classification task. The result is a **multiple logistic regression** with the following probability:

$p(C_1, X) \approx \frac{e^{\beta_0 + \sum_i^{} \beta_iX_i}}{1 + e^{\beta_0 +  \sum_i^{} \beta_iX_i}}$

In the following example, we use the features petal width and length. With these two feature, the classifier computes the probability that a new flower is *Iris virginica". 
The black dashed line it the decision boundary of the model with a probability of 50%. Each of the other parallel lines represent different probabilities computed by the model.  
"""
)

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


st.subheader("Softmax Regression")

st.markdown(
    r"""
In the previous examples, we whated to know wheter a certain sample belongs to a certain class or not. But what if we want to know whether is belongs to a certain class
or another one? Or even more then that. Well, then we have to use **Softmax Regression** with the following softmax function. 

$p(C_k, X) \approx \frac{e^{\beta_kX}}{1 + \sum_i^{} e^{\beta_jX}}$

* $k$ is the number of classes
* $\beta_X$ is a vector conteining the scores of each class for instance x
* $p(C_k, X)$ is the estimated probability that an instance x belongs to klass k, given the scores of each class for that instance

For now, we won't go into more detail in this topic here. But we'll might in the future. 😉
"""
)

# Page 148
