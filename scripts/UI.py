import os
import sys
sys.path.append(os.path.abspath('..'))
import streamlit as st
import pickle
import numpy as np
from src import config

st.set_page_config(
    page_title="Real Estate Price Prediction",  # Titolo della pagina
    page_icon="🏠",  # Favicon
)

st.markdown("""
    <style>
        /* Personalizza il titolo */
        h1 {
            color: #ff0000 !important;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Real Estate Price Prediction")

#model_name = st.selectbox("Select Model",("Random Forest", "Logistic Regression"),)
#model_path = f"{config.MODELS_PATH}random_forest.pickle" if model_name == "Random Forest" else f"{config.MODELS_PATH}logistic_regression.pickle"
model_path = f"{config.MODELS_PATH}linear_regression.pickle"
model_name = "Linear Regression"

if not os.path.exists(model_path):
    st.error(f"No trained model found for {model_name}. Run the training script first.")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)  

user_lat = st.number_input("Insert latitude:")
user_long = st.number_input("Insert longitude:")
user_inputs = [user_lat, user_long]

if st.button("Predict"):
    if any(field == 0.00 for field in user_inputs):
        st.warning("Please fill in all fields.")
    else:
        # transform input and predict
        X = np.array([[float(value) for value in user_inputs]])
        prediction = round(model.predict(X)[0], 2)
        st.success(f"Predicted price: {prediction} €")
        