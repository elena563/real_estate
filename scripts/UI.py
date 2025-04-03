import os
import sys
sys.path.append(os.path.abspath('..'))
import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Real Estate Price Prediction",  # page title
    page_icon="🏠",  # favicon
)
st.markdown("""
    <style>
        h1 {
            color: #ff4b4b !important;
            text-align: center;
        }
        .stButton{
            display: flex;
            justify-content: center;
        }
        .st-emotion-cache-ocsh0s, .st-emotion-cache-ocsh0s:focus, .st-emotion-cache-ocsh0s:focus:not(:active) {
            background-color: #ff4b4b;
            border-color: #ff4b4b;
            color: white;
            width: 200px;
        }
        .st-emotion-cache-ocsh0s:hover{
            background-color: white;
            border-color: #ff4b4b;
            color: #ff4b4b;
        }
        .st-emotion-cache-ocsh0s:focus-visible {
            box-shadow: none;
        }
        .st-emotion-cache-1dj3ksd {
            background-color: #ff4b4b;
        }
        .st-emotion-cache-mtjnbi {
            padding-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)
st.title("How much is worth a house in Taiwan? Find out now!")
st.text("Enter your real estate details in Sindian, New Taipei, and let a machine learning model predict the price per square meter for you!")

MODELS_PATH = 'linear_regression.pickle'

if not os.path.exists(MODELS_PATH):
    st.error(f"No trained model found. Run the training script first.")
else:
    with open(MODELS_PATH, "rb") as f:
        model1 = pickle.load(f)
        model2 = pickle.load(f)  

if "input_option" not in st.session_state:
    st.session_state.input_option = "coordinates"

col1, col2 = st.columns(2)
with col1:
    if st.button("Coordinates"):
        st.session_state.input_option = "coordinates"
with col2:
    if st.button("More Stats"):
        st.session_state.input_option = "more"

if st.session_state.input_option == "coordinates":
    st.subheader("Coordinates")
    lat = st.number_input("At what latitude is it?", value=None, max_value=25.01459, min_value=24.93207, placeholder="Type a value between 24,93 and 25,01")
    long = st.number_input("At what longitude is it?", value=None, max_value=121.56627, min_value=121.47353, placeholder="Type a value between 121,47 and 121,56")
    user_inputs = [lat, long]
    model = model1
    
elif st.session_state.input_option == "more":
    st.subheader("More Stats")
    age = st.number_input("How old is it?", min_value=0, max_value=100, value=None, placeholder="Type age in years...")
    dist = st.number_input("At what distance is the nearest MRT station?", min_value=0, value=None, placeholder="Type distance in meters...")
    stores = st.slider("How many convenience stores are there nearby?", 0, 10)
    user_inputs = [age, dist, stores]
    model = model2


if st.button("Predict"):
    if any(field == None for field in user_inputs):
        st.warning("Please fill in all fields.")
    elif any(field < 0 for field in user_inputs):
        st.warning("All variables need to be positive.")
    else:
        X = np.array([[float(value) for value in user_inputs]])
        prediction = round(model.predict(X)[0], 2)
        st.success(f"Predicted price: {prediction} € per m\u00B2")
        