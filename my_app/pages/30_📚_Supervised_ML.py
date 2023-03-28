import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

st.title("Supervised Machine Learning")

st.markdown(
    r"""
Supervised machine learning is a type of machine learning algorithm that involves training a model on a labeled dataset to predict outcomes 
or classifications for new, unseen data. In supervised learning, the dataset used for training the model includes both input variables (features) 
and output variables (labels or targets), where the goal is to learn a mapping between the input variables and the corresponding output variables.

The process of supervised learning typically involves the following steps:

* Data collection: Collect a labeled dataset that includes input and output variables.
* Data preprocessing: Clean, normalize, and transform the data to prepare it for modeling.
* Model selection: Select a suitable model architecture and algorithm to train on the dataset.
* Training: Train the model on the labeled dataset to learn the mapping between the input and output variables.
* Evaluation: Evaluate the performance of the trained model on a separate validation dataset to assess its accuracy and generalization ability.
* Prediction: Use the trained model to predict outcomes or classifications for new, unseen data.

Supervised learning can be used for a wide range of applications, such as image classification, object detection, speech recognition, 
natural language processing, and predictive modeling in finance and healthcare. The success of supervised learning depends on the quality and 
quantity of the labeled dataset used for training the model, as well as the selection of appropriate model architecture and hyperparameters.
"""
)

