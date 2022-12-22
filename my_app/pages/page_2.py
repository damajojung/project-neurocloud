# This is the second page of the streamlit app.

import streamlit as st
import numpy as np
import pandas as pd
import os

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


data_load_state = st.text("Loading data...")
dat = load_data(data_dir)
data_load_state.text("Done! (using st.cache)")

# Show data
st.dataframe(dat)

# Display plot
