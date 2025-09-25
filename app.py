# app/app.py
import streamlit as st
import pandas as pd
import sys, os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import preprocessing and model functions
from src.preprocess import preprocess_data
from src.model import train_model, save_model

st.title("Customer Segmentation Dashboard")

# File uploader (CSV or Excel)
uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    # Read file based on extension
    file_name = uploaded_file.name
    if file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type!")
        st.stop()

    st.write("Preview of uploaded file", df.head())

    # Preprocess data
    df_processed, scaler, label_encoders = preprocess_data(df)

    # Train model on this dataset
    model = train_model(df_processed, algo="kmeans", n_clusters=5)

    # Save model and scaler for future use
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_model(model, scaler,
               model_path=os.path.join(project_root, "models", f"{file_name}_model.pkl"),
               scaler_path=os.path.join(project_root, "models", f"{file_name}_scaler.pkl"))

    st.success("Model trained and saved successfully!")

    # Scale and predict clusters
    X_scaled = scaler.transform(df_processed)
    clusters = model.predict(X_scaled)
    df['Cluster'] = clusters

    st.write("Clustered Data", df.head())
    st.bar_chart(df['Cluster'].value_counts())

else:
    st.warning("Please upload a CSV or Excel file first!")
