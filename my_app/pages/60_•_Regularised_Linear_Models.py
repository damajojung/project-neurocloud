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
def get_figure(x, y, data_frame):

    fig, ax = plt.subplots(1, figsize=(5, 3))
    ax.scatter(x, y, s=7)

    ax.set_xlabel("GDP per capita (USD)")
    ax.set_ylabel("Life satisfaction")
    return fig, ax


# Display Data
X = np.array(dat.GDP)
y = np.array(dat.Satisfaction)

fig, ax = get_figure(X, y, dat)

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
Regularised Linear Regression.
"""
)

st.header("Ridge Regression")

st.header("Lasso Regression")
