import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib # For saving the scaler

from data_handler import fetch_data, preprocess_data

def build_lstm_model(input_shape):
    """Builds the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, model_path='spy_lstm_model.h5', scaler_path='scaler.gz'):
    """Trains the LSTM model and saves it."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = build_lstm_model((X_train.shape[1], 1))

    # Callbacks
    # Reduced patience for faster early stopping in this re-run for main branch.
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50, # Reduced epochs for faster re-run for main branch
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    print("Model training complete.")
    return model, history

if __name__ == '__main__':
    # 1. Fetch and preprocess data
    print("Fetching and preprocessing data...")
    spy_data = fetch_data()
    X, y, scaler = preprocess_data(spy_data, sequence_length=60)
    print(f"Data shapes: X: {X.shape}, y: {y.shape}")

    # Save the scaler
    scaler_filename = 'spy_scaler.gz'
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")

    # 2. Train the model
    # Using .keras extension as it's preferred
    trained_model, training_history = train_model(X, y, model_path='spy_lstm_model.keras', scaler_path=scaler_filename)

    print("\nTo make a prediction for tomorrow:")
    print("1. Get the last 'sequence_length' days of 'Close' prices.")
    print("2. Scale these prices using the saved 'spy_scaler.gz'.")
    print("3. Reshape the scaled data to (1, sequence_length, 1).")
    print("4. Use `loaded_model.predict()` on this data.")
    print("5. Inverse transform the prediction using `loaded_scaler.inverse_transform()`.")

    if len(X) > 0:
        last_sequence = X[-1].reshape(1, X.shape[1], 1)
        prediction_scaled = trained_model.predict(last_sequence)
        prediction_actual = scaler.inverse_transform(prediction_scaled)
        print(f"\nExample - Predicted next 'Close' (scaled): {prediction_scaled[0][0]}")
        print(f"Example - Predicted next 'Close' (actual): {prediction_actual[0][0]}")
    else:
        print("\nNot enough data to make a test prediction.")
