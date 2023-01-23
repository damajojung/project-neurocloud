import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

# Configure the appearance
import matplotlib as mpl
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


plt.style.use(["default"])

mpl.rc("axes", labelsize=8)  # Labels of the axes
mpl.rc("xtick", labelsize=8)  # x-ticks
mpl.rc("ytick", labelsize=8)  # y-ticks

st.set_page_config(layout="wide")

# Get data
dir = os.getcwd()
data_dir = dir + "/app_data/sat_gdp.csv"


@st.cache
def load_data(path):
    data = pd.read_csv(path, index_col=0)
    return data


dat = load_data(data_dir)

# Creating function for getting plots
def get_figure(x, y):

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.scatter(x, y, s=7)

    ax.set_xlabel("GDP per capita (USD)")
    ax.set_ylabel("Life satisfaction")
    return fig, ax


# Display Data
X = np.array(dat.GDP)
y = np.array(dat.Satisfaction)

fig, ax = get_figure(X, y)

position_text = {"Ireland": (65000, 6.0), "Luxembourg": (95000, 6.0)}

for country, pos_text in position_text.items():
    pos_data_x, pos_data_y = (
        float(dat[dat["Country"] == country]["GDP"]),
        float(dat[dat["Country"] == country]["Satisfaction"]),
    )
    country = "U.S." if country == "United States" else country
    plt.annotate(
        country,
        xy=(pos_data_x, pos_data_y),
        xytext=pos_text,
        arrowprops=dict(facecolor="black", width=0.5, shrink=0.1, headwidth=5),
    )
    plt.plot(pos_data_x, pos_data_y, "ro")

st.pyplot(fig)


st.title("Regularised Linear Regression")

st.markdown(
    """
Regularise polynomial models can be achieved by reducing the number of polynomial degrees. Linear models on the other hand can be regularised by 
constraining the weights of the model. This helps to avoid overfitting the data Regularisation forces the model not only to fit the data but also keep the weights as small
 as possible. 
Two version of regularised linear models will be discussed in this section, namely the Ridge Regression (also Tikhonov regularisation) and
the Lasso Regression. 
"""
)

#########################
# Machine Learning
#########################

# I don't think that this section is useful anymore.
# X = np.array(dat.GDP)
# y = np.array(dat.Satisfaction)

# # Select a linear model
# model = sklearn.linear_model.LinearRegression(fit_intercept=True)

# # Train the model
# model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

# x_fit = np.linspace(min(X), max(X), 41)
# y_fit = model.predict(x_fit[:, np.newaxis])

# # Create figure
# fig, ax = get_figure(X, y)
# plt.plot(x_fit, y_fit, c="red", linewidth=0.5, label="Regression Line")
# ax.legend(loc="best")
# st.pyplot(fig)

#########################


st.header("Ridge Regression")

st.markdown(
    r"""
The goal is still to find the coefficients $b_0, ... , b_n$ that minimize the residual sum of squares. However, a reularised term is added 
to the cost function which is called 'shrinkage penalty' and looks as follows:
$$\alpha \sum^{m}_{i = 1} b_i^2$$ 
"""
)

st.markdown(
    r"""
We can see that it is still quadratic. Therefore, it is still possible to find a closed form solution. It should be noted that the shrinkage 
penalty should only be added during training. Once the model is trained, one should use the unregularised performance measure to evaluate
the performance of the model. The hyperparameter $\alpha$ controls how much one wants to regularise the model. Setting it to $\alpha = 0$ 
results in a Linear Regression. Big $\alpha$ values result in a flat line which goes through the
mean of the data. The following eqauation shows the whole cost function for the Ridge Regression:

$J(\theta) = MSE(\theta) + \alpha * \frac{1}{2} \sum^{n}_{i = 1} b_i ^2$

Tell something about 1/2
"""
)

x_fit = np.linspace(min(X), max(X), 41)


def plot_model(model_class, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = (
            model_class(10 ** alpha, **model_kargs) if alpha > 0 else LinearRegression()
        )
        model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
        y_new_regul = model.predict(x_fit.reshape(-1, 1))
        lw = 2 if alpha > 0 else 1
        plt.plot(
            x_fit,
            y_new_regul,
            style,
            linewidth=lw,
            label=r"$\alpha = {}$" + "10^" + f"{alpha}",
        )
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)


fig, ax = get_figure(X, y)
plot_model(Ridge, alphas=(0, 9.8, 10.3), random_state=42)

st.pyplot(fig)

st.header("Lasso Regression")

st.markdown(
    r"""
Lasso stands for 'Least Absolute Shrinkage and Selection Operator' and is another regularised version of Linear Regression. A regularisation term is added to
the cost function. This time, the $l_1$ norm of the weight vector is used insted of the square ($l-2$ norm) in ridge regression. The Lasso Regression cost
function looks as follows:

$J(\theta) = MSE(\theta) + \alpha * \frac{1}{2} \sum^{n}_{i = 1} | b_i | $

It has to be mentioned that an interesting aspect of Lasso Regression is that there is a tendency to eliminate the weights of the least important feature by setting it to 
zero which results in a feature selection and thus to a sparse model. 
"""
)

x_fit = np.linspace(min(X), max(X), 41)

ridge_reg = Lasso(alpha=1, random_state=42)
ridge_reg.fit(X.reshape(-1, 1), y.reshape(-1, 1))

fig1, ax1 = get_figure(X, y)
plot_model(Lasso, alphas=(0, 3, 3.7), random_state=42)

st.pyplot(fig1)


st.markdown(
    r"""
The effect of the regularisation becomes more visible with higher degrees. On the left hand side, we can see the Ridge regression and on the right
hand side the Lasso regression with different $\alpha$. In genereal, they behave similar with the given data. It is clearly visible that a bigger
$\alpha$ leads to a smoother regression line. 
"""
)

np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = (
    (X ** 5)
    + (X ** 4)
    - 2 * (X ** 3)
    - 10 * (X ** 2)
    + (3 * X)
    + np.random.randn(m, 1)
    + np.random.normal(0, 40, m).reshape(m, 1)
)
X_new = np.linspace(-3, 3, 100).reshape(100, 1)


def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline(
                [
                    (
                        "poly_features",
                        PolynomialFeatures(degree=17, include_bias=False),
                    ),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ]
            )
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(
            X_new,
            y_new_regul,
            style,
            linewidth=lw,
            label=r"$\alpha = {}$".format(alpha),
        )
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)


fig2 = plt.figure(figsize=(8, 4))
plt.subplot(121)
plot_model(Ridge, polynomial=True, alphas=(0, 1, 100), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 0.5, 1), random_state=42)

st.pyplot(fig2)
