from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib # For loading the scaler
import os
from data_handler import preprocess_data, fetch_data # Import functions from data_handler

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'spy_lstm_model.keras'
SCALER_PATH = 'spy_scaler.gz'
SEQUENCE_LENGTH = 60  # Must match the sequence_length used during training
PREDICTION_DAYS = 1 # Number of days to predict into the future

# --- Load Model and Scaler ---
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            model = None
            scaler = None
    else:
        print("Model or scaler file not found. Please train the model first by running model.py.")

load_model_and_scaler() # Load on startup

@app.route('/', methods=['GET'])
def index():
    if model is None or scaler is None:
        return "Model not loaded. Please ensure 'spy_lstm_model.keras' and 'spy_scaler.gz' exist and the app has permission to read them. You might need to run model.py first.", 500

    prediction_actual = None
    error_message = None
    last_actual_close = None
    prediction_date = None

    try:
        # Fetch the latest 100 days of data to ensure enough for sequence_length
        today = pd.to_datetime('today')
        start_date_fetch = (today - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
        latest_data_full = fetch_data(ticker="SPY", start_date=start_date_fetch, end_date=today.strftime('%Y-%m-%d'))

        if latest_data_full.empty or len(latest_data_full) < SEQUENCE_LENGTH:
            error_message = "Not enough historical data to make a prediction."
        else:
            last_actual_close_price = latest_data_full['Close'].iloc[-1]
            last_actual_close_date = latest_data_full.index[-1].strftime('%Y-%m-%d')
            last_actual_close = f"${last_actual_close_price:,.2f} on {last_actual_close_date}"

            # Get the last SEQUENCE_LENGTH days of close prices
            close_prices = latest_data_full['Close'].values.reshape(-1, 1)

            # Scale the recent data using the loaded scaler
            # IMPORTANT: Only transform, do not fit again
            scaled_prices = scaler.transform(close_prices)

            # Take the last sequence_length points
            last_sequence_scaled = scaled_prices[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)

            # Predict the next day
            prediction_scaled = model.predict(last_sequence_scaled)

            # Inverse transform the prediction to get the actual price
            prediction_actual_price = scaler.inverse_transform(prediction_scaled)[0][0]
            prediction_actual = f"${prediction_actual_price:,.2f}"

            # Determine the prediction date
            # Assuming the last data point is for "today" or the last trading day
            # and we are predicting the next trading day.
            # This is a simplification; true next trading day logic can be complex.
            prediction_date = (latest_data_full.index[-1] + pd.Timedelta(days=PREDICTION_DAYS)).strftime('%Y-%m-%d')


    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(f"Error during prediction: {e}") # Log to console as well

    return render_template('index.html', prediction=prediction_actual, error=error_message, last_close=last_actual_close, prediction_date=prediction_date)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') # Make it accessible on the network
```
