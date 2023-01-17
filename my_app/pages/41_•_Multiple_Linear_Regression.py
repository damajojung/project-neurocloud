import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import linear_model

st.set_page_config(layout="wide")

st.title("Multiple Linear Regression")

st.markdown(
    r"""
In section $Regression$, we used one feature variable to make a prediction. However, we can use as many feature variables as we want as it is shown in the following
equation:

$ \hat{y} = b_0 + b_1 * x_1 + b_2 * x_2 + ... + b_n * x_n $

"""
)


st.markdown(
    r"""
Let us assume that there is a certain bush in California which does not like hot weather. Therefore, it can be found in the mountains 
where the temperatures are lower thoughout the year. Scientist have collected data about the sea level, the height of the bush, the 
size of the leafs and the concentration of magnesium and caesium in the soil 1 meter apart from the roots. The data can be seen in the following data frame.
"""
)

plt.style.use(["default"])

# To plot pretty figures directly within Jupyter
import matplotlib as mpl

mpl.rc("axes", labelsize=10)
mpl.rc("xtick", labelsize=10)
mpl.rc("ytick", labelsize=10)

# Creating data

m = 0.008
c = 6  # data with approximate straight line
x = np.linspace(50, 2000, 100)
y = m * x + c + np.random.randn(100) * 2 + 150
z = m * x + c + np.random.randn(100) * 1 + 10

# Bad data
y_1 = m * x + c + np.random.randn(100) * 10  # looks random
y_2 = (m * (x ** 3) + x) / 1000  # Not linear

dat = pd.DataFrame(
    {"Sealevel": x, "Height": y, "Leafs": z, "Magnesium": y_1, "Caesium": y_2}
)

X = pd.DataFrame({"sealevel": x, "height": y}).values.reshape(-1, 2)

### End of Data creation

# Show data frame
st.dataframe(dat)

# Create data for prediction
x_pred = np.linspace(0, 2000, 20)  # range of porosity values
y_pred = np.linspace(155, 172.5, 20)  # range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

# Show data as figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

ax.scatter(x, y, z)
ax.set_xlabel("Sea level (m)")
ax.set_ylabel("Height (cm)")
ax.set_zlabel("Size of leafs (mm)")

st.pyplot(fig)

# Compare the variables

st.markdown(
    r"""
Since we deal only with 3 variables, we can plot this data nicely. However, if you are dealing with more than three values, one has to use a visualisation
and check, which vairables make sense to take into consideration. This can be seen in the following plot. 
"""
)

# Compare all the variables

fig = plt.figure(figsize=(5, 5))

fig = sns.pairplot(dat)

st.pyplot(fig)

st.markdown(
    r"""
It is clearly visible that all the variables share a linear relationship. Therefore, it makes sense to use all of them. However, if you are dealing with hundrets
of variables, one has to use a faster way to determine which variables are worthwhile using and which are not. 
"""
)
# Prediction

ols = linear_model.LinearRegression()
model = ols.fit(X, z)

predicted = model.predict(model_viz)

r2 = model.score(X, z)

### Show data with prediction surface

fig1 = plt.figure(figsize=(15, 5))

ax1 = fig1.add_subplot(131, projection="3d")
ax2 = fig1.add_subplot(132, projection="3d")
ax3 = fig1.add_subplot(133, projection="3d")

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.scatter(x, y, z)
    ax.scatter(
        xx_pred.flatten(),
        yy_pred.flatten(),
        predicted,
        facecolor=(0, 0, 0, 0),
        s=10,
        edgecolor="olive",
    )
    ax.set_xlabel("Sea level (m)", fontsize=10)
    ax.set_ylabel("Height (cm)", fontsize=10)
    ax.set_zlabel("Size of leafs (mm)", fontsize=10)
    ax.locator_params(nbins=4, axis="x")
    ax.locator_params(nbins=5, axis="x")

fig1.suptitle("$R^2 = %.2f$" % r2, fontsize=30)

ax1.view_init(elev=27, azim=112)
ax2.view_init(elev=24, azim=-51)
ax3.view_init(elev=60, azim=165)

st.pyplot(fig1)

#################################### Getting statistics

import statsmodels.api as sm

# add constant to predictor variables
x = sm.add_constant(X)

# fit linear regression model
model = sm.OLS(y, X).fit()

# view model summary
summary = model.summary()

st.markdown(
    r"""
The p-values of each variable must be < .05. Therefore, both vairables work in this case. 
"""
)

st.text(f"{summary}")

################################### Getting VIF

# Checking for the VIF values of the variables.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Creating a dataframe that will contain the names of all the feature variables and their VIFs
vif = pd.DataFrame()
vif["Features"] = pd.DataFrame(X).columns
vif["VIF"] = [
    variance_inflation_factor(pd.DataFrame(X).values, i)
    for i in range(pd.DataFrame(X).shape[1])
]
vif["VIF"] = round(vif["VIF"], 2)
vif = vif.sort_values(by="VIF", ascending=False)

st.markdown(
    r"""
The VIF of each variable must be < 5. So both variables work in this case. 
"""
)
st.text(f"{vif}")
