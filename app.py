# app.py - Streamlit dashboard for load monitoring and prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pickle
import datetime

# Load artifacts
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_explainer():
    with open("shap_explainer.pkl", "rb") as f:
        return pickle.load(f)

# Main app
st.set_page_config(page_title="Smart Grid Load Monitoring", layout="wide")
st.title("⚡ Real-Time Load Monitoring and Prediction Dashboard")

# Upload data
uploaded_file = st.file_uploader("smart_grid_dataset.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp').sort_index()

    st.success("✅ Data loaded successfully!")
    st.dataframe(df.tail())

    # Feature engineering (based on training logic)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['rolling_mean_24h'] = df['Power Consumption (kW)'].rolling(window=24).mean()
    df = df.dropna()

    # Predict
    model = load_model()
    scaler = load_scaler()
    explainer = load_explainer()

    X = df[['hour', 'dayofweek', 'rolling_mean_24h']]
    scaled_X = scaler.transform(X)
    df['Predicted Load'] = model.predict(scaled_X)

    # Anomaly detection: define anomaly as actual load > predicted + 2*std
    residual = df['Power Consumption (kW)'] - df['Predicted Load']
    threshold = residual.mean() + 2 * residual.std()
    df['Anomaly'] = np.where(np.abs(residual) > threshold, 1, 0)

    # Plots
    st.subheader("📈 Actual vs Predicted Load")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['Power Consumption (kW)'], label='Actual', color='blue')
    ax.plot(df.index, df['Predicted Load'], label='Predicted', color='orange')
    ax.set_title("Actual vs Predicted Load")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🚨 Detected Anomalies")
    anomaly_df = df[df['Anomaly'] == 1]
    st.write(f"{len(anomaly_df)} anomalies detected.")
    st.dataframe(anomaly_df[['Power Consumption (kW)', 'Predicted Load']])

    # SHAP explainability
    st.subheader("🧠 SHAP Feature Importance")
    shap_values = explainer(scaled_X, check_additivity=False)
    fig_shap = shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(bbox_inches='tight', dpi=300)

    st.subheader("🔍 SHAP Bar Plot")
    fig_bar = shap.plots.bar(shap_values, show=False)
    st.pyplot(bbox_inches='tight', dpi=300)

else:
    st.info("Please upload a CSV file containing smart grid load data.")
