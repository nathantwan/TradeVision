from flask import Flask, jsonify, request
from model import train_model, load_trained_model, predict_stock_price
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Set up logging to help with debugging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return "Welcome to the Stock Prediction API!"

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    ticker = data.get('ticker')
    load_existing_model = data.get('load_existing_model', False)  # Whether to load an existing model

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        # Call the train_model function
        train_model(ticker, load_existing_model=load_existing_model)
        return jsonify({"message": f"Model trained successfully for ticker: {ticker}"}), 200
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Preflight request received'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    try:
        # Step 1: Check if the request contains JSON data
        if not request.is_json:
            logging.error("Request data is not in JSON format")
            return jsonify({'error': 'Request must be in JSON format'}), 400

        # Step 2: Get the JSON data from the request
        data = request.get_json()

        # Step 3: Check if the 'ticker' key exists in the incoming request data
        ticker = data.get('ticker', None)
        if not ticker:
            logging.error("Ticker is missing in the request data")
            return jsonify({'error': 'Ticker is missing'}), 400

        # Step 4: Check if the 'time' key exists and is a valid number
        time = data.get('time', None)
        if not time or not isinstance(time, int) or time <= 0:
            logging.error("Time is missing or invalid")
            return jsonify({'error': 'Time is missing or invalid'}), 400

        logging.info(f"Received request to predict stock price for ticker: {ticker} for {time} days ahead")

        # Step 5: Load the trained model
        model = load_trained_model()
        if model is None:
            logging.error("Model could not be loaded")
            return jsonify({'error': 'Model could not be loaded'}), 500

        # Step 6: Make the prediction for the specified number of days ahead
        prediction = predict_stock_price(model, ticker, time)

        if not prediction:
            logging.error(f"Prediction failed for ticker: {ticker}")
            return jsonify({'error': 'Prediction failed'}), 500

        # Step 7: Return the prediction as a response
        logging.info(f"Prediction successful: {prediction}")
        return jsonify({'predictions': prediction})

    except Exception as e:
        # Log the error if something goes wrong
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Function to start the Flask app
if __name__ == '__main__':
    app.run(debug=True)