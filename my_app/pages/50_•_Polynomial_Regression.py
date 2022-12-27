import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import linear_model

st.set_page_config(layout="wide")

st.title("Polynomial Regression")

st.markdown(
    """
This app performs a linear regression from GDP per capita against the happiness index.
* **Python libraries:** base64, pandas, streamlit, sci-kit learn
* **Data source:** [satisfaction.com](https://stats.oecd.org/index.aspx?DataSetCode=BLI#).
* **Data source:** [GDP.com](https://data.oecd.org/gdp/gross-domestic-product-gdp.htm).
"""
)

# Settings
plt.style.use(["default"])

# To plot pretty figures directly within Jupyter
import matplotlib as mpl

mpl.rc("axes", labelsize=12)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# Creating function for getting plots
def get_fig():

    fig, ax = plt.subplots(1, figsize=(5, 3))

    return fig, ax


# Generate Data
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Plot data
fig, ax = get_fig()

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])

st.pyplot(fig)

# Machine Learning
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]

X_poly[0]


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
fig1, ax1 = get_fig()
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
st.pyplot(fig1)


# Comparison
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

fig2, ax2 = get_fig()
plt.plot(X, y, "b.")

for style, width, degree in (("g-", 1, 100), ("b--", 2, 2), ("r-.", 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline(
        [
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ]
    )
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

plt.legend(loc="upper left")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
st.pyplot(fig2)
