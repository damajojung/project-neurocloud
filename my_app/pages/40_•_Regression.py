# This is the second page of the streamlit app.

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

st.title("Regression")

st.markdown(
    """
Let's begin with one of the simplest models there is, namely the linear regression.
This app performs a linear regression using GDP per capita to predict the stisfaction index of a certain country.
Both datasets are taken from the OECD website. But let's start with the theory behind it. 
* **Python libraries:** base64, numpy, pandas, streamlit, sklearn, matplotlib, seaborn
* **Data source satisfaction index:** [oecd-satisfaction-index.com](https://stats.oecd.org/index.aspx?DataSetCode=BLI#).
* **Data source GDP score:** [oecd-gdp-per-capita.com](https://data.oecd.org/gdp/gross-domestic-product-gdp.htm).
"""
)

st.subheader("Theory")

st.markdown(
    r"""
In simple terms, a linear regression tries to identify whether there is a connection between two or more variables. This can be written as follows:

$ \hat{y} = b_0 + b_1 * x_1 + b_2 * x_2 + ... + b_n * x_n $

Where:
* $\hat{y}$ is the predicted value
* $n$ is the number of features
* $x_i$ is the $i^{th}$ feature value
* $b_i$ is the $j^{th}$ model parameter including the bias term $b_0$ and the feature weights $b_1$, $b_2$ etc.

In this example, we get the following equation:

$ life\_satisfaction\_index = b_0 + b_1 * GDP\_per capita $
"""
)

st.subheader("Training")

st.markdown(
    r"""
All we have to do right now is to train the model. In order to do that, one needs a measurement of how well or poorly the model fits the training data. The most common performance measurement used in regression is the Root Mean Square Error (RMSE) which can be seen in the following equation:

$RMSE(\bf{X},h) = \sqrt{ \frac{1}{m} \sum^{m}_{i = 1} (h(\bf{x}^{(i)}) - y^{(i)} )^2}$

"""
)

st.markdown(
    r"""
Even though the RMSE is most of the times the status quo measurement when it come to regression, it might make more sense to take another one under certain conditions. For example, if the data contains a large number of outliers. In such a case, one might consider using the Mean Absolute Error (MAE) which can be seen in the following equation:

$MAE(\bf{X},h) = \frac{1}{m} \sum^{m}_{i = 1} | h(\bf{x}^{(i)}) - y^{(i)} |$

"""
)

# Check out, how to only make the x bold in the equaitons. Now, everything is bold.

# Get data
dir = os.getcwd()
data_dir = dir + "/app_data/sat_gdp.csv"


@st.cache
def load_data(path):
    data = pd.read_csv(path, index_col=0)
    return data


dat = load_data(data_dir)

# Show data
dat = dat.sort_values(by=["LOCATION"], ascending=True)
st.dataframe(dat)

# Creating function for getting plots
def get_figure(x, y, data_frame):

    fig, ax = plt.subplots(1, figsize=(5, 3))
    ax.scatter(x, y, s=7)

    for i, txt in enumerate(data_frame.LOCATION):
        ax.annotate(txt, (x[i], y[i]), size=5, c="black")

    ax.set_xlabel("GDP per capita (USD)")
    ax.set_ylabel("Life satisfaction")
    return fig, ax


# Sidebar - Countries selection
st.markdown(
    """
Remove or add some countries of interest by either clicking the x on the card or clicking into the field of cards whereupon a dropdown menue appears.
* The graph will be updated accordingly.
* It is recommended to unselect LUX and IRL which results in a clearly visible change. 
"""
)
locs_sorted = sorted(dat.LOCATION.unique())
selected_location = st.multiselect(
    "Countries",
    locs_sorted,
    [
        "AUS",
        "BEL",
        "CAN",
        "CZE",
        "DNK",
        "FIN",
        "FRA",
        "GRC",
        "ISL",
        "IRL",
        "ITA",
        "JPN",
        "KOR",
        "LUX",
        "MEX",
        "NZL",
        "NOR",
        "POL",
        "PRT",
        "SVK",
        "CHE",
        "TUR",
        "CHL",
        "ISR",
        "RUS",
        "ZAF",
        "COL",
        "CRI",
    ],
)

# Filtering data
df_selected_locs = dat[(dat.LOCATION.isin(selected_location))]

#########################
# Machine Learning
#########################

X = np.array(df_selected_locs.GDP)
y = np.array(df_selected_locs.Satisfaction)

# Select a linear model
model = sklearn.linear_model.LinearRegression(fit_intercept=True)

# Train the model
model.fit(X.reshape(-1, 1), y.reshape(-1, 1))

# Get Model Parameters
t0, t1 = (
    str(np.round(model.intercept_[0], 2)),
    str(model.coef_[0][0])[:4] + str(model.coef_[0][0])[-4:],
)

# Make a prediction for Cyprus
cyprus_gdp_per_capita = 22587
X_new = [[cyprus_gdp_per_capita]]  # Cyprus' GDP per capita
cyprus_sat_pred = model.predict(X_new)

x_fit = np.linspace(min(X), max(X), 1000)
y_fit = model.predict(x_fit[:, np.newaxis])


# Create figure
fig, ax = get_figure(X, y, df_selected_locs)
plt.plot(x_fit, y_fit, c="red", linewidth=0.5, label="Regression Line")
plt.scatter(cyprus_gdp_per_capita, cyprus_sat_pred, c="green", s=7)
plt.plot(
    [cyprus_gdp_per_capita, cyprus_gdp_per_capita],
    [0, cyprus_sat_pred],
    "g--",
    linewidth=0.5,
    label=f"Prediction Cyprus",
)
plt.text(35000, 3.0, f"$\tSatisfaction = {t0} + GDP * {t1}$", fontsize=8, color="r")
plt.text(
    35000,
    4.0,
    f"$\tCyprus Prediction  = {np.round(cyprus_sat_pred[0],2)[0]}$",
    fontsize=8,
    color="g",
)
ax.legend(loc="best")
st.pyplot(fig)

