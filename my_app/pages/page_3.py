# This is the third page of the streamlit app.

import streamlit as st
import numpy as np
import pandas as pd

st.title("Page 3")

st.markdown(
    """
This app performs a linear regression from GDP per capita against the happiness index.
* **Python libraries:** base64, pandas, streamlit, sci-kit learn
* **Data source:** [satisfaction.com](https://stats.oecd.org/index.aspx?DataSetCode=BLI#).
* **Data source:** [GDP.com](https://data.oecd.org/gdp/gross-domestic-product-gdp.htm).
"""
)

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.line_chart(chart_data)

