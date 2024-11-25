import streamlit as st
import numpy as np
import pickle

# Title and Description
st.title("Model Deployment with Streamlit")
st.write("Upload inputs and get predictions!")

# Load the Model
with open('model_pickle', 'rb') as model_file:
    model = pickle.load(model_file)

# Input Form
st.sidebar.header("Input Features")
def user_input_features():
    feature_1 = st.sidebar.number_input("Area", min_value=0.0, max_value=10000.0, step=100.0, value=0.0)
    feature_2 = st.sidebar.number_input("Bedrooms", min_value=0.0, max_value=10.0, step=1.0, value=0.0)
    feature_3 = st.sidebar.number_input("Age", min_value=0.0, max_value=100.0, step=1.0, value=0.0)
    return np.array([feature_1, feature_2, feature_3]).reshape(1, -1)


inputs = user_input_features()

# Prediction
if st.button("Predict"):
    prediction = model.predict(inputs)
    st.success(f"Prediction: {round(prediction[0])}")

# Footer
st.write("---")
st.write("Streamlit Example Deployment")