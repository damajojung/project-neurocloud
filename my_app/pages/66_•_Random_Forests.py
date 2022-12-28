import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sklearn
from sklearn import datasets
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(layout="wide")

st.title("Random Forests")

st.markdown(
    """
Random Forests.
"""
)


# Get data
dir = os.getcwd()
data_dir = dir + "/app_data/possum.csv"


@st.cache
def load_data(path):
    data = pd.read_csv(path)
    return data


df = load_data(data_dir)
df = df.dropna()

X = df.drop(["case", "site", "Pop", "earconch", "footlgth", "sex"], axis=1)
y = df["sex"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=500, max_features="auto", random_state=42
)
rf_model.fit(X_train, y_train)


predictions = rf_model.predict(X_test)
comparison = pd.DataFrame({"Predictions": predictions, "Actual": np.array(y_test)})
st.dataframe(comparison)

cm = confusion_matrix(y_test, predictions, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)

st.pyplot(fig)

importances = rf_model.feature_importances_
columns = X.columns
i = 0

while i < len(columns):
    st.caption(
        f"The importance of feature {columns[i]} is {round(importances[i] * 100,2)} %."
    )
    i += 1
