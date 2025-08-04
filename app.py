import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Configure layout to prevent scrolling
st.set_page_config(page_title="Fire Type Prediction", layout="centered")

# Load model and encoders
model = joblib.load('fire_type_model.pkl')
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# App Title
st.title("🔥 Fire Type Prediction")

st.markdown("Fill the inputs to predict the **fire type** based on MODIS data.")

# Two-column layout for compact view
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", format="%.4f")
    longitude = st.number_input("Longitude", format="%.4f")
    brightness = st.number_input("Brightness")
    scan = st.number_input("Scan")
    track = st.number_input("Track")
    acq_time = st.number_input("Acquisition Time", step=1)

with col2:
    satellite = st.selectbox("Satellite", label_encoders['satellite'].classes_)
    instrument = st.selectbox("Instrument", label_encoders['instrument'].classes_)
    confidence = st.slider("Confidence", 0, 100, step=1)
    bright_t31 = st.number_input("Brightness T31")
    frp = st.number_input("Fire Radiative Power")
    daynight = st.selectbox("Day/Night", label_encoders['daynight'].classes_)

st.markdown("---")

# Predict Button
if st.button("🚀 Predict Fire Type"):
    # Encode categorical fields
    satellite_encoded = label_encoders['satellite'].transform([satellite])[0]
    instrument_encoded = label_encoders['instrument'].transform([instrument])[0]
    daynight_encoded = label_encoders['daynight'].transform([daynight])[0]

    # Form input for prediction
    input_data = np.array([[latitude, longitude, brightness, scan, track,
                            acq_time, satellite_encoded, instrument_encoded,
                            confidence, bright_t31, frp, daynight_encoded]])

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"🔥 **Predicted Fire Type: {prediction}**")
