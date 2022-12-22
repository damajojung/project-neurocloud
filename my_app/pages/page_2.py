# This is the second page of the streamlit app.

import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configure the appearance
import matplotlib as mpl
import seaborn as sns

mpl.rc("axes", labelsize=5)
mpl.rc("xtick", labelsize=5)
mpl.rc("ytick", labelsize=5)

st.set_page_config(layout="wide")

plt.style.use(["ggplot"])

st.title("Page 2 - Regression")

st.markdown(
    """
This app performs a linear regression from GDP per capita against the stisfaction index from the OECD website.
* **Python libraries:** base64, pandas, streamlit, sci-kit learn
* **Data source:** [oecd-satisfaction-index.com](https://stats.oecd.org/index.aspx?DataSetCode=BLI#).
* **Data source:** [oecd-gdp-per-capita.com](https://data.oecd.org/gdp/gross-domestic-product-gdp.htm).
"""
)

# Get data
dir = os.getcwd()
data_dir = dir + "/app_data/sat_gdp.csv"


@st.cache
def load_data(path):
    data = pd.read_csv(path, index_col=0)
    return data


dat = load_data(data_dir)

# Show data
st.dataframe(dat)

# Visualize the data
X = np.array(dat.GDP)
y = np.array(dat.Satisfaction)

fig, ax = plt.subplots(1, figsize=(5, 3))
ax.scatter(X, y, s=5)

for i, txt in enumerate(dat.LOCATION):
    ax.annotate(txt, (X[i], y[i]), size=4, c="black")

ax.set_xlabel("GDP per capita (USD)")
ax.set_ylabel("Life satisfaction")
plt.show()

# Display plot
st.pyplot(fig)
