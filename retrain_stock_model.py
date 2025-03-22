import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, save_model
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from datetime import datetime, timedelta
import os
import glob
import tensorflow as tf

# Enable eager execution
tf.config.run_functions_eagerly(True)

def retrain_stock_model(base_model_path, target_ticker, epochs=8, time_step=100, model_folder='stock-models'):
    """
    Retrain a pre-trained stock model for a specific ticker.
    
    Parameters:
      - base_model_path: Path to the pre-trained base model.
      - target_ticker: Stock ticker symbol to fine-tune for.
      - epochs: Number of training epochs.
      - time_step: Sequence length for the LSTM.
      - model_folder: Folder to store all fine-tuned models.
    
    Returns:
      - A dict with training history, test loss, ticker, and model path on success.
      - False if retraining fails.
    """
    # Check if base model exists
    if not os.path.exists(base_model_path):
        print(f"Base model not found at {base_model_path}")
        return False

    # Ensure the model folder exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Get yesterday's date as end date
    yesterday = datetime.now() - timedelta(days=1)
    end_date = yesterday.strftime('%Y-%m-%d')

    # Download stock data for the target ticker
    print(f"Downloading data for {target_ticker}...")
    stock_data = yf.download(target_ticker, start='2014-02-21', end=end_date)

    if len(stock_data) < 200:
        print(f"Not enough data for {target_ticker}")
        return False

    # Prepare data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    # Create dataset sequences for the LSTM
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    X, y = np.array(X), np.array(y)

    # Split data into training and testing sets
    train_size = 0.8
    X_train = X[:int(X.shape[0] * train_size)]
    X_test = X[int(X.shape[0] * train_size):]
    y_train = y[:int(y.shape[0] * train_size)]
    y_test = y[int(y.shape[0] * train_size):]

    # Reshape for LSTM: [samples, time_steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Load the base model
    print(f"Loading base model from {base_model_path}...")
    model = load_model(base_model_path)

    # Recompile to reinitialize the optimizer’s state
    model.compile(optimizer='adam', loss='mse')

    # Fine-tune model on target ticker's data
    print(f"Fine-tuning model for {target_ticker}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print(f"Test loss after fine-tuning: {loss}")

    # Prepare today's model filename
    today_str = datetime.now().strftime("%Y-%m-%d")
    fine_tuned_filename = f"{target_ticker}_fine_tuned_model_{today_str}.h5"
    fine_tuned_path = os.path.join(model_folder, fine_tuned_filename)

    # Delete older models for the same stock (if any)
    pattern = os.path.join(model_folder, f"{target_ticker}_fine_tuned_model_*.h5")
    old_models = glob.glob(pattern)
    for old_model in old_models:
        if old_model != fine_tuned_path:
            print(f"Deleting older model: {old_model}")
            os.remove(old_model)

    # Save the fine-tuned model
    model.save(fine_tuned_path)
    print(f"Fine-tuned model saved to {fine_tuned_path}")

    return {
        'ticker': target_ticker,
        'model_path': fine_tuned_path,
        'history': history.history,
        'test_loss': loss
    }

# Example usage
if __name__ == "__main__":
    # Path to the pre-trained base model on the Canadian market
    base_model_path = "canadian_market_base_model.h5"
    # Ticker symbol to fine-tune for
    target_ticker = "BN-PFG.TO"

    result = retrain_stock_model(
        base_model_path=base_model_path,
        target_ticker=target_ticker,
        epochs=8,
        time_step=100,
        model_folder='stock-models'
    )

    if result:
        print(f"Successfully retrained model for {result['ticker']}")
        print(f"Model saved to {result['model_path']}")
        print(f"Final test loss: {result['test_loss']}")
    else:
        print("Retraining failed")