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
    """
Let us assume that there is a certain bush in California which does not like hot weather. Therefore, it can be found in the mountains 
where the temperatures are lower thoughout the year. Scientist have collected data about the sea level, the height of the bush and the 
size of the leafs. The data can be seen in the following data frame.
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

dat = pd.DataFrame({"Sealevel": x, "Height": y, "Leafs": z})

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
