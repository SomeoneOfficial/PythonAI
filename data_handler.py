import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker="SPY", start_date="2010-01-01", end_date=None):
    """Fetches historical stock data."""
    if end_date is None:
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data, sequence_length=60):
    """Preprocesses the data for LSTM model."""
    # Use 'Close' price for prediction
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(len(scaled_prices) - sequence_length):
        X.append(scaled_prices[i:i+sequence_length, 0])
        y.append(scaled_prices[i+sequence_length, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape X for LSTM [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

if __name__ == '__main__':
    # Example usage
    spy_data = fetch_data()
    print("Raw data head:")
    print(spy_data.head())

    X_processed, y_processed, data_scaler = preprocess_data(spy_data)
    print("\nProcessed X shape:", X_processed.shape)
    print("Processed y shape:", y_processed.shape)
    print("First processed X sample:", X_processed[0])
    print("First processed y sample:", y_processed[0])
    print("\nScaler:", data_scaler)
