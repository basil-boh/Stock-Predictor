import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, ticker, start_date=None, end_date=None, prediction_days=60):
        """
        Initialize the StockPredictor with a stock ticker and date range.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            prediction_days (int): Number of days to use for prediction
        """
        self.ticker = ticker
        self.prediction_days = prediction_days
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
            
        self.start_date = start_date
        self.end_date = end_date
        
        # Load data
        self.data = self._load_data()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def _load_data(self):
        """Load stock data from Yahoo Finance."""
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return df
    
    def prepare_data(self):
        """Prepare data for training."""
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))
        
        # Prepare training data
        x_train, y_train = [], []
        
        for i in range(self.prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i-self.prediction_days:i, 0])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train
    
    def build_model(self, x_train):
        """Build and compile the LSTM model."""
        model = Sequential()
        
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model
        return model
    
    def train(self, epochs=25, batch_size=32):
        """Train the model."""
        if self.model is None:
            x_train, y_train = self.prepare_data()
            self.build_model(x_train)
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        else:
            x_train, y_train = self.prepare_data()
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def predict_next_days(self, days=30):
        """Predict stock prices for the next specified days."""
        # Get the last prediction_days of data
        test_data = self.data['Close'][-self.prediction_days:].values.reshape(-1, 1)
        test_data = self.scaler.transform(test_data)
        
        predictions = []
        current_batch = test_data.copy()
        
        for _ in range(days):
            # Prepare the input data
            current_batch_reshaped = current_batch.reshape((1, self.prediction_days, 1))
            
            # Get the predicted price
            predicted_price = self.model.predict(current_batch_reshaped)
            
            # Add the prediction to our list
            predictions.append(predicted_price[0, 0])
            
            # Update the batch by removing the first value and adding the prediction
            current_batch = np.append(current_batch[1:], predicted_price)
            current_batch = current_batch.reshape(-1, 1)
        
        # Inverse transform to get actual prices
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions
    
    def plot_predictions(self, days=30):
        """Plot the historical data and predictions."""
        predictions = self.predict_next_days(days)
        
        # Generate dates for predictions
        last_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
        
        # Plot historical data
        plt.figure(figsize=(16, 8))
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        
        # Plot historical data
        plt.plot(self.data.index, self.data['Close'], label='Historical Data')
        
        # Plot predictions
        plt.plot(future_dates, predictions, label='Predicted Price')
        
        plt.legend()
        plt.show()
        
        return predictions, future_dates