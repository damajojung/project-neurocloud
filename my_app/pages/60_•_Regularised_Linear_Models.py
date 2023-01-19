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
constrining the weights of the model. Regularisation forces the model not only to fit the data but also keep the weights as small as possible. 
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
$$\alpha \sum^{m}_{i = 1} \beta^2$$ 
"""
)

st.markdown(
    r"""
We can see that it is still quadratic. Therefore, it is still possible to find a closed form solution. It should be noted that the shrinkage 
penalty should only be added during training. Once the model is trained, one should use the unregularised performance measure to evaluate
the performance of the model. The hyperparameter $\alpha$ controls how much one wants to regularise the model. Setting it to $\alpha = 0$ 
results in a Linear Regression. The bigger $\alpha$, the closer the weights come to zero which results in a flat line going through the
mean of the data. The following eqauation shows the whole cost function for the Ridge Regression:

$J(\beta) = MSE(\beta) + \alpha * \frac{1}{2} \sum^{n}_{i = 1} \alpha_i ^2$

Tell something about 1/2
"""
)

from sklearn.linear_model import Ridge

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


from sklearn.linear_model import Lasso

x_fit = np.linspace(min(X), max(X), 41)

ridge_reg = Lasso(alpha=1, random_state=42)
ridge_reg.fit(X.reshape(-1, 1), y.reshape(-1, 1))

fig1, ax1 = get_figure(X, y)
plot_model(Lasso, alphas=(0, 3, 3.7), random_state=42)

st.pyplot(fig1)
