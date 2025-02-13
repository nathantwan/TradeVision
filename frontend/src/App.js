import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [ticker, setTicker] = useState(''); // State for stock ticker
  const [time, setTime] = useState(30); // State for number of days to predict
  const [predictions, setPredictions] = useState([]); // State to store predictions
  const [loading, setLoading] = useState(false); // State for loading indicator
  const [error, setError] = useState(''); // State for error messages

  // Function to handle form submission
  const handlePredict = async (e) => {
    e.preventDefault(); // Prevent default form submission behavior

    // Validate input
    if (!ticker || !time) {
      setError('Please enter a valid ticker and time.');
      return;
    }

    setLoading(true); // Show loading indicator
    setError(''); // Clear any previous errors

    try {
      // Make a POST request to the Flask backend
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        ticker: ticker.toUpperCase(), // Convert ticker to uppercase
        time: parseInt(time), // Ensure time is an integer
      });

      // Update predictions state with the response data
      setPredictions(response.data.predictions);
    } catch (err) {
      // Handle errors
      setError('Failed to fetch predictions. Please try again.');
      console.error(err);
    } finally {
      setLoading(false); // Hide loading indicator
    }
  };

  return (
    <div className="App">
      <h1>Stock Price Prediction</h1>
      <form onSubmit={handlePredict}>
        <div>
          <label htmlFor="ticker">Stock Ticker:</label>
          <input
            type="text"
            id="ticker"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="Enter ticker (e.g., AAPL)"
          />
        </div>
        <div>
          <label htmlFor="time">Days to Predict:</label>
          <input
            type="number"
            id="time"
            value={time}
            onChange={(e) => setTime(e.target.value)}
            min="1"
          />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {/* Display error message if any */}
      {error && <p className="error">{error}</p>}

      {/* Display predictions */}
      {predictions.length > 0 && (
        <div className="predictions">
          <h2>Predictions:</h2>
          <ul>
            {predictions.map((pred, index) => (
              <li key={index}>
                <strong>{pred.prediction_date}:</strong> ${pred.prediction.toFixed(2)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;