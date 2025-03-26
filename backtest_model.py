# backtest_model.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf

# Ensure eager execution is enabled.
tf.config.run_functions_eagerly(True)

def backtest_model_accuracy(model, scaler, stock_data, time_step=100, forecast_horizon=30):
    """
    Backtest the model on the last part of historical data using a sliding window.
    Returns RMSE and MAPE (in percentage).

    Parameters:
      - model: the loaded Keras model.
      - scaler: the MinMaxScaler used for training.
      - stock_data: a DataFrame containing historical data with a 'Close' column.
      - time_step: number of past days to use for prediction.
      - forecast_horizon: number of days over which to backtest.
    """
    # Use the last (forecast_horizon + time_step) days for backtesting.
    test_values = stock_data['Close'].values[-(forecast_horizon + time_step):]
    scaled_data = scaler.transform(test_values.reshape(-1, 1)).flatten()

    predictions = []
    actuals = []

    # Clear any lingering session context
    tf.keras.backend.clear_session()

    for i in range(time_step, len(scaled_data) - 1):
        window = scaled_data[i - time_step : i]
        window = window.reshape(1, time_step, 1)
        pred_scaled = float(model.predict(window, verbose=0)[0][0])
        predictions.append(pred_scaled)
        actuals.append(scaled_data[i])

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    return rmse, mape