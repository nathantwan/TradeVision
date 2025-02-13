import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


# Function to fetch stock data
def fetch_stock_data(ticker, start_date='2010-01-01', end_date='2025-01-01'):
   try:
       stock_data = yf.download(ticker, start=start_date, end=end_date)
       if stock_data.empty:
           raise ValueError(f"No data found for ticker: {ticker}")
       return stock_data
   except Exception as e:
       print(f"Error fetching data for {ticker}: {e}")
       return None




# Preprocess data for LSTM model
def preprocess_data(stock_data):
   data = stock_data[['Close']].values
   scaler = MinMaxScaler(feature_range=(0, 1))
   data_scaled = scaler.fit_transform(data)


   # Prepare the data for LSTM (lookback of 60 days)
   X, y = [], []
   for i in range(60, len(data_scaled)):
       X.append(data_scaled[i-60:i, 0])  # Take 60 days as input features
       y.append(data_scaled[i, 0])  # Use the next day's price as target


   X, y = np.array(X), np.array(y)


   # Reshape X for LSTM input (3D: samples, time steps, features)
   X = np.reshape(X, (X.shape[0], X.shape[1], 1))


   # Log shapes to check if dimensions are correct
   print(f"X shape: {X.shape}")
   print(f"y shape: {y.shape}")


   return X, y, scaler


# Build the LSTM model
def build_model():
   model = Sequential()
   model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
   model.add(LSTM(units=50, return_sequences=False))
   model.add(Dense(units=1))  # Output layer
   model.compile(optimizer='adam', loss='mean_squared_error')
   return model


# Train the model
def train_model(ticker):
   # Fetch the data
   stock_data = fetch_stock_data(ticker)
  
   # Preprocess the data
   X, y, scaler = preprocess_data(stock_data)
  
   # Build the model
   model = build_model()


   # Train the model
   model.fit(X, y, epochs=20, batch_size=32)


   # Save the model
   model.save('stock_model.keras')
   return model, scaler


def load_trained_model():
   try:
       model = load_model('stock_model.keras')
       print("Model loaded successfully.")
       return model
   except Exception as e:
       print(f"Error loading model: {e}")
       return None


# Function to make a prediction using the trained model
def predict_stock_price(model, ticker, time):
   stock_data = fetch_stock_data(ticker)
   X, _, scaler = preprocess_data(stock_data)  # You only need X and scaler for prediction


   # Get the latest data for prediction (the last 60 days)
   latest_data = X[-1:]


   # Log the shape of latest_data to check consistency
   print(f"latest_data shape before prediction: {latest_data.shape}")


   # Initialize a list to store predictions and the corresponding dates
   predictions = []
   prediction_dates = []


   # Loop to predict for the number of days specified
   for i in range(time):
       # Predict the next day's price
       prediction_scaled = model.predict(latest_data)


       # Inverse transform the prediction to get the actual price
       prediction = scaler.inverse_transform(prediction_scaled)


       # Add the prediction to the list
       predictions.append(prediction[0][0])


       # Get the date of the prediction (the latest date in the data + 1 day)
       prediction_date = stock_data.index[-1] + timedelta(days=i+1)
       prediction_dates.append(prediction_date.strftime('%Y-%m-%d'))


       # Update the latest data to include the predicted value (reshape properly)
       prediction_scaled_reshaped = prediction_scaled.reshape(1, 1, 1)
       latest_data = np.append(latest_data[:, 1:, :], prediction_scaled_reshaped, axis=1)


       # Log the shape after update to check if it's correct
       print(f"latest_data shape after update: {latest_data.shape}")


   # Convert prediction_dates to datetime objects
   prediction_dates = pd.to_datetime(prediction_dates)


   # Plotting the results
   plt.figure(figsize=(10, 6))


   # Plot the actual data (last 'time' days) and predictions
   actual_data = stock_data[['Close']].tail(time)  # Last 'time' days of actual data
   plt.plot(actual_data.index, actual_data['Close'], color='blue', label='Actual Closing Prices')


   plt.plot(prediction_dates, predictions, color='red', label='Predicted Prices')


   # Adding labels and title
   plt.xlabel('Date')
   plt.ylabel('Stock Price (USD)')
   plt.title(f'{ticker} Stock Price Prediction')
   plt.legend()


   # Display the plot
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.show()


   # Prepare the response with the predictions and additional details
   response = []
   for i in range(time):
       response.append({
           'prediction': float(predictions[i]),  # Convert to float for JSON serialization
           'ticker': ticker,
           'prediction_date': prediction_dates[i].strftime('%Y-%m-%d'),
           'price_type': 'Closing Price',  # Add any type of price you're predicting, like 'Closing Price'
       })


   return response