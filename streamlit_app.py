import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta
import os

# Import the retraining function from your retrain file.
from retrain_stock_model import retrain_stock_model

# Set page configuration
st.set_page_config(
    page_title="Canadian Stock Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Canadian Stock Market Forecaster")
st.markdown("""
This application uses a two-stage deep learning model to forecast stock prices for Canadian stocks.
If a stock's model is not yet trained for today’s data or is missing, it will be retrained automatically.
""")

class StockForecaster:
    def __init__(self, time_step=100, base_model_path='canadian_market_base_model.h5'):
        self.time_step = time_step
        self.base_model_path = base_model_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.daily_volatility = None

    def download_data(self, ticker, start_date='2014-02-21', end_date=None):
        if end_date is None:
            yesterday = datetime.now() - timedelta(days=1)
            end_date = yesterday.strftime('%Y-%m-%d')
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if len(data) < 200:
                return None
            return data
        except Exception as e:
            st.error(f"Error downloading {ticker}: {e}")
            return None

    def prepare_data(self, stock_data):
        if stock_data is None or len(stock_data) == 0:
            return None, None

        # Calculate historical daily volatility
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        self.daily_volatility = stock_data['Daily_Return'].std()

        # Scale the closing prices
        scaled_data = self.scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
        return stock_data, scaled_data

    def load_or_retrain_model(self, ticker):
        """
        For Canadian stocks, check if a fine-tuned model exists in 'stock-models'
        with today’s date. If not, retrain from the base model.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        model_folder = "stock-models"
        model_filename = f"{ticker}_fine_tuned_model_{today_str}.h5"
        model_path = os.path.join(model_folder, model_filename)
        
        # If the model file doesn't exist, retrain
        if not os.path.exists(model_path):
            st.info(f"Retraining model for {ticker} for date {today_str}...")
            retrain_result = retrain_stock_model(
                base_model_path=self.base_model_path,
                target_ticker=ticker,
                epochs=8,
                time_step=self.time_step,
                model_folder=model_folder
            )
            if not retrain_result:
                st.error("Retraining failed. Using base model instead.")
                if os.path.exists(self.base_model_path):
                    self.model = load_model(self.base_model_path, compile=False)
                    return True
                else:
                    return False
            else:
                self.model = load_model(model_path, compile=False)
                return True
        else:
            st.success(f"Loading up-to-date fine-tuned model for {ticker}")
            self.model = load_model(model_path, compile=False)
            return True

    def generate_forecast(self, stock_data, scaled_data, forecast_days=180):
        if self.model is None:
            st.error("No model available. Please ensure the model is loaded or retrained.")
            return None

        # Get the last sequence for prediction
        last_sequence = scaled_data[-self.time_step:].reshape(self.time_step)
        curr_seq = last_sequence.copy()
        predicted_prices = []

        with st.spinner('Generating forecast...'):
            for _ in range(forecast_days):
                curr_seq_reshaped = curr_seq.reshape(1, self.time_step, 1)
                next_pred = float(self.model.predict(curr_seq_reshaped, verbose=0)[0][0])
                # Optionally add noise based on historical volatility
                daily_vol = float(self.daily_volatility) if self.daily_volatility is not None else 0.01
                noise_factor = 1 + np.random.normal(0, daily_vol * 0.7)
                noisy_pred = next_pred * noise_factor
                # Constrain changes to a reasonable range
                noisy_pred = max(min(noisy_pred, curr_seq[-1] * 1.15), curr_seq[-1] * 0.85)
                predicted_prices.append(noisy_pred)
                curr_seq = np.append(curr_seq[1:], noisy_pred)
        
        predicted_array = np.array(predicted_prices).reshape(-1, 1)
        try:
            base_forecast = self.scaler.inverse_transform(predicted_array).flatten()
        except Exception as e:
            st.error(f"Error scaling forecast data: {e}")
            return None
        return base_forecast

    def calculate_forecast_metrics(self, ticker, stock_data, base_forecast):
        last_date = stock_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(base_forecast))
        current_price = float(stock_data['Close'].iloc[-1])
        forecast_price = float(base_forecast[-1])
        potential_return = (forecast_price / current_price - 1) * 100

        metrics = {
            'ticker': ticker,
            'current_price': current_price,
            'forecast_date': forecast_dates[-1].strftime('%Y-%m-%d'),
            'forecast_price': forecast_price,
            'potential_return': potential_return
        }
        return metrics

# List of Canadian stock tickers only
canadian_tickers = [
    'AC.TO', 'RY.TO', 'TD.TO', 'ENB.TO', 'SU.TO', 'BMO.TO', 'CP.TO',
    'BNS.TO', 'BCE.TO', 'CM.TO', 'TRP.TO', 'MFC.TO', 'T.TO', 'FTS.TO',
    'QSR.TO', 'SLF.TO', 'CCO.TO', 'L.TO', 'CNQ.TO', 'ATD.TO', 'AQN.TO', 'PPL.TO'
]

st.sidebar.header("Forecast Settings")
selected_ticker = st.sidebar.selectbox("Select Canadian Stock Ticker", options=canadian_tickers, index=0,
                                         help="Choose a Canadian stock to forecast")
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=30, max_value=365,
                                     value=180, step=30, help="Number of days to forecast")

if st.sidebar.button("Generate Forecast"):
    forecaster = StockForecaster()
    
    # Ensure the base model exists
    if not os.path.exists(forecaster.base_model_path):
        st.error("Base model file 'canadian_market_base_model.h5' not found in the current directory.")
        st.stop()
    
    with st.spinner(f"Analyzing {selected_ticker}..."):
        stock_data = forecaster.download_data(selected_ticker)
        if stock_data is None:
            st.error(f"Not enough data for {selected_ticker}")
            st.stop()
        stock_data, scaled_data = forecaster.prepare_data(stock_data)
    
    # Load or retrain the fine-tuned model
    if not forecaster.load_or_retrain_model(selected_ticker):
        st.error("Model loading or retraining failed.")
        st.stop()
    
    # Generate forecast (base forecast only)
    base_forecast = forecaster.generate_forecast(stock_data, scaled_data, forecast_days=forecast_horizon)
    if base_forecast is None:
        st.error("Forecast generation failed.")
        st.stop()
    
    # Prepare interactive chart data
    historical_dates = stock_data.index
    historical_prices = stock_data['Close'].values.flatten()  # Ensure 1D array
    forecast_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    
    df_hist = pd.DataFrame({'Historical Prices': historical_prices}, index=historical_dates)
    df_forecast = pd.DataFrame({'6-Month Forecast': base_forecast}, index=forecast_dates)
    df_combined = pd.concat([df_hist, df_forecast])
    st.line_chart(df_combined)
    
    # Calculate and display forecast metrics
    metrics = forecaster.calculate_forecast_metrics(selected_ticker, stock_data, base_forecast)
    st.header("Forecast Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Current Price", value=f"${metrics['current_price']:.2f}")
        st.metric(label=f"Forecast Price ({metrics['forecast_date']})", value=f"${metrics['forecast_price']:.2f}",
                  delta=f"{metrics['potential_return']:.2f}%")
    with col2:
        st.markdown(f"""
        **Ticker:** {metrics['ticker']}  
        **Forecast Date:** {metrics['forecast_date']}
        """)
else:
    st.info("👈 Select a stock and click 'Generate Forecast' to begin analysis")
    st.markdown("""
    ### How to use this application:
    
    1. Select a Canadian stock ticker from the dropdown menu.  
    2. Adjust the forecast horizon (number of days to forecast).  
    3. Click "Generate Forecast" to run the analysis.
    
    The application will automatically retrain the model if it hasn't been updated for today.
    """)

st.markdown("---")
st.markdown("### Model Information")
st.markdown("""
This application uses:
- A base model trained on multiple Canadian stocks (`canadian_market_base_model.h5`).  
- Fine-tuned models for individual stocks stored in the **stock-models** folder (retrained daily).  
- LSTM neural networks for time series forecasting.
""")