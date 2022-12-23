import streamlit as st
import numpy as np
import pandas as pd

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

