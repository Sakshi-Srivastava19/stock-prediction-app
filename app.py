import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------------- Load Models ----------------------
lr_model = joblib.load("models/linear_regression.joblib")
ann_model = load_model("models/ann_model.h5", compile=False)
scaler = joblib.load("models/minmax_scaler.joblib")

# ---------------------- Prediction Logic ----------------------
def predict(df, model_name):
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    if model_name == "Linear Regression":
        preds = lr_model.predict(X_scaled)
    elif model_name == "ANN":
        preds = ann_model.predict(X_scaled).reshape(-1)

    df["Predicted"] = preds
    return df

# ---------------------- Sidebar Navigation ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Prediction"])

# ---------------------- HOME PAGE ----------------------
if page == "Home":
    st.title("📊 Stock Price Prediction App")
    st.write("""
    Welcome to the Stock Price Prediction App!

    Use the sidebar to navigate:  
    - **Dashboard**: Explore historical stock data  
    - **Prediction**: Predict future stock prices using ML models
    """)
    st.image("https://static.streamlit.io/examples/dice.jpg", width=400)

# ---------------------- DASHBOARD PAGE ----------------------
# ---------------------- DASHBOARD PAGE ----------------------
elif page == "Dashboard":
    st.title("📈 Stock Dashboard")
    
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

    if st.button("Load Data"):
        st.info("Fetching stock data...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                st.error("No data found. Please check the ticker and date range.")
            else:
                st.success(f"Data for {ticker.upper()} loaded!")
                df.reset_index(inplace=True)
                st.dataframe(df.head(10))

                # Moving Averages
                st.subheader("Moving Averages")
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA50'] = df['Close'].rolling(50).mean()
                df['MA200'] = df['Close'].rolling(200).mean()

                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(df['Date'], df['Close'], label="Close")
                ax.plot(df['Date'], df['MA20'], label="MA20")
                ax.plot(df['Date'], df['MA50'], label="MA50")
                ax.plot(df['Date'], df['MA200'], label="MA200")
                ax.legend()
                st.pyplot(fig)


        except Exception as e:
            st.error(f"Error fetching data: {e}")


# ---------------------- PREDICTION PAGE ----------------------
elif page == "Prediction":
    st.title("🔮 Stock Price Prediction")
    st.write("Predict stock prices using Linear Regression or ANN.")

    ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, INFY):", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))
    model_choice = st.selectbox("Select Model:", ["Linear Regression", "ANN"])

    if st.button("Fetch & Predict"):
        st.info("Fetching data...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found.")
        else:
            df.reset_index(inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            st.success("Data fetched!")

            st.info(f"Running predictions using {model_choice}...")
            result_df = predict(df, model_choice)
            st.success("Prediction complete!")

            # Plot Actual vs Predicted
            st.subheader("📈 Actual vs Predicted Closing Prices")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(result_df["Date"], result_df["Close"], label="Actual Price")
            ax.plot(result_df["Date"], result_df["Predicted"], label="Predicted Price")
            ax.legend()
            st.pyplot(fig)

            # Show table
            st.subheader("📄 Prediction Table")
            st.dataframe(result_df.tail())

            # Download CSV
            csv = result_df.to_csv(index=False).encode()
            st.download_button("⬇ Download Predictions CSV", csv, "predictions.csv", "text/csv")
