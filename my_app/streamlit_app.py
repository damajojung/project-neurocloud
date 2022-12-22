# This is the main page of the streamlit app.

import streamlit as st
import numpy as np
import pandas as pd

st.title("Introduction")

st.markdown(
    """
Welcome to my webpage where I present the wonderful world of machine and deep learning. The saphire sun wishes a pleasant journey!
* **Python libraries:** base64, pandas, numpy,  streamlit, sci-kit learn
* **Data source:** [satisfaction.com](https://stats.oecd.org/index.aspx?DataSetCode=BLI#).
* **Data source:** [GDP.com](https://data.oecd.org/gdp/gross-domestic-product-gdp.htm).
"""
)

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.line_chart(chart_data)

