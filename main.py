import argparse
from stock_predictor import StockPredictor

def main():
    parser = argparse.ArgumentParser(description='AI Stock Price Predictor')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--prediction_days', type=int, default=60, help='Number of days to use for prediction')
    parser.add_argument('--future_days', type=int, default=30, help='Number of days to predict into the future')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = StockPredictor(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_days=args.prediction_days
    )
    
    print(f"Training model for {args.ticker}...")
    predictor.train(epochs=args.epochs, batch_size=args.batch_size)
    
    print(f"Predicting next {args.future_days} days...")
    predictions, dates = predictor.plot_predictions(days=args.future_days)
    
    print("\nPredicted prices for the next days:")
    for date, price in zip(dates, predictions):
        print(f"{date.strftime('%Y-%m-%d')}: ${price[0]:.2f}")

if __name__ == "__main__":
    main()