import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import talib  # For calculating technical indicators

MODEL_PATH = "stock_model.keras"
SCALER_PATH = "scaler.pkl"

# Function to fetch stock data
def fetch_stock_data(ticker, start_date="2010-01-01", end_date="2025-01-01"):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data = stock_data.astype(np.float64)

        if stock_data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        if len(stock_data) < 60:
            raise ValueError(f"Not enough data for {ticker}. Need at least 60 days of data.")
        
        # Flatten the MultiIndex columns (if present)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)  # Remove the ticker level
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Preprocess data for LSTM model
def preprocess_data(stock_data, save_scaler=False):
    # Ensure 'Close' and 'Volume' are 1-dimensional arrays
    close_prices = stock_data['Close'].values  # Convert to numpy array (1D)
    volume = stock_data['Volume'].values  # Convert to numpy array (1D)

    # Calculate technical indicators
    stock_data['RSI'] = talib.RSI(close_prices, timeperiod=14)  # RSI
    stock_data['MACD'], stock_data['MACD_signal'], stock_data['MACD_hist'] = talib.MACD(
        close_prices, fastperiod=12, slowperiod=26, signalperiod=9  # MACD
    )
    stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = talib.BBANDS(
        close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0  # Bollinger Bands
    )
    stock_data['OBV'] = talib.OBV(close_prices, volume)  # On-Balance Volume

    # Drop rows with NaN values (caused by indicator calculations)
    stock_data.dropna(inplace=True)

    # Select relevant features
    features = stock_data[['Close', 'RSI', 'MACD', 'MACD_hist', 'Upper_Band', 'Lower_Band', 'OBV']].values

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    if save_scaler:
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    X, y = [], []
    for i in range(60, len(features_scaled)):
        X.append(features_scaled[i-60:i, :])  # Use all features
        y.append(features_scaled[i, 0])  # Predict 'Close' price

    X, y = np.array(X), np.array(y)
    print(f"X shape: {X.shape}")  # Should be (samples, 60, num_features)
    print(f"y shape: {y.shape}")  # Should be (samples,)


    if len(X) == 0 or len(y) == 0:
        raise ValueError("Preprocessing resulted in empty datasets.")

    return X, y, scaler

# Build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),  # Input shape: (time_steps, num_features)
        LSTM(units=50, return_sequences=False),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Train the modelx
def train_model(ticker, load_existing_model=False):
    # Fetch data for the new ticker
    stock_data = fetch_stock_data(ticker)
    if stock_data is None:
        return None, None

    X, y, _ = preprocess_data(stock_data, save_scaler=True)
    input_shape = (X.shape[1], X.shape[2])  # (time_steps, num_features)
    print(f"Input shape: {input_shape}")  # Debugging: Print input shape

    # Load the existing model or build a new one
    if load_existing_model:
        try:
            model = load_model(MODEL_PATH)
            print(f"Loaded existing model for fine-tuning on ticker: {ticker}")
        except Exception as e:
            print(f"Error loading model: {e}. Building a new model.")
            model = build_model(input_shape)
    else:
        print("Building a new model.")
        model = build_model(input_shape)

    # Fine-tune the model
    try:
        early_stopping = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        print("Starting model training...")
        history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        print("Model training completed.")
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

    return model

# Load the trained model
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the saved scaler
def load_scaler():
    try:
        with open(SCALER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

# Function to make a stock price prediction
def predict_stock_price(model, ticker, days=30):
    # Fetch stock data
    stock_data = fetch_stock_data(ticker)
    if stock_data is None or model is None:
        return None

    # Preprocess the data and fetch the scaler
    X, _, scaler = preprocess_data(stock_data)

    if scaler is None:
        # Load the scaler if it's not passed from preprocess
        scaler = load_scaler()
        if scaler is None:
            print("Error: No scaler found. Cannot proceed with prediction.")
            return None

    # Use the latest data for predictions
    latest_data = X[-1:]  # Shape (1, 60, 7)

    predictions = []
    prediction_dates = []

    # Predict for the specified number of days
    for i in range(days):
        # Predict the scaled closing price
        prediction_scaled = model.predict(latest_data)  # Shape (1, 1)

        # Create a placeholder array with the same number of features as the scaler was trained on
        placeholder_features = np.zeros((1, 7))  # Shape (1, 7)
        placeholder_features[0, 0] = prediction_scaled[0, 0]  # Set the first feature (Close) to the predicted value

        # Inverse transform to get actual price values
        prediction = scaler.inverse_transform(placeholder_features)  # Shape (1, 7)
        predictions.append(prediction[0, 0])  # Extract the inverse-transformed Close price

        # Calculate the prediction date for the future day
        prediction_date = stock_data.index[-1] + timedelta(days=i+1)
        prediction_dates.append(prediction_date.strftime('%Y-%m-%d'))

        # Update the latest data with the new prediction for the next iteration
        # Reshape the prediction and append it to the latest data
        prediction_scaled_reshaped = np.zeros((1, 1, 7))  # Shape (1, 1, 7)
        prediction_scaled_reshaped[0, 0, 0] = prediction_scaled[0, 0]  # Set the first feature (Close)

        # Append the new predicted value to the sequence of previous features (closing price)
        # Use np.append to update the `latest_data` sequence, maintaining shape (1, 60, 7)
        latest_data = np.append(latest_data[:, 1:, :], prediction_scaled_reshaped, axis=1)

    # Convert prediction dates to pandas datetime objects
    prediction_dates = pd.to_datetime(prediction_dates)

    # Plot the results
    plt.figure(figsize=(10, 6))
    actual_data = stock_data[['Close']].tail(days)
    plt.plot(actual_data.index, actual_data['Close'], color='blue', label='Actual Closing Prices')
    plt.plot(prediction_dates, predictions, color='red', label='Predicted Prices')

    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("prediction_plot.png")  # Save plot instead of showing
    print("Prediction plot saved as 'prediction_plot.png'.")

    # Prepare the response to return in API-compatible format
    response = [
        {
            "prediction": float(predictions[i]),
            "ticker": ticker,
            "prediction_date": prediction_dates[i].strftime('%Y-%m-%d'),
            "price_type": "Closing Price"
        }
        for i in range(days)
    ]

    return response
