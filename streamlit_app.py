import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go

# Import the retraining function
from retrain_stock_model import retrain_stock_model

# --- UI Setup ---
st.set_page_config(page_title="Canadian Stock Forecast", page_icon="📈", layout="wide")
st.title("Canadian Stock Forecast")

# --- Forecaster Class ---
class StockForecaster:
    def __init__(self, time_step=100, base_model_path='canadian_market_base_model.h5'):
        self.time_step = time_step
        self.base_model_path = base_model_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.daily_volatility = None
        self.test_loss = None  # Store test loss if retrained

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
        # Compute historical volatility
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        self.daily_volatility = stock_data['Daily_Return'].std()
        # Scale data
        scaled_data = self.scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
        return stock_data, scaled_data

    def load_or_retrain_model(self, ticker):
        """
        Load an up-to-date fine-tuned model for a Canadian stock if it exists.
        Otherwise, retrain from the base model and store in the "stock-models" folder.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        model_folder = "stock-models"
        model_filename = f"{ticker}_fine_tuned_model_{today_str}.h5"
        model_path = os.path.join(model_folder, model_filename)
        
        if not os.path.exists(model_path):
            st.info(f"Retraining model for {ticker} ({today_str})...")
            result = retrain_stock_model(
                base_model_path=self.base_model_path,
                target_ticker=ticker,
                epochs=8,
                time_step=self.time_step,
                model_folder=model_folder
            )
            if not result:
                st.error("Retraining failed. Loading base model.")
                if os.path.exists(self.base_model_path):
                    self.model = load_model(self.base_model_path, compile=False)
                    self.test_loss = None
                    return True
                return False
            else:
                self.test_loss = result.get("test_loss", None)
                self.model = load_model(model_path, compile=False)
                return True
        else:
            st.success(f"Using up-to-date model for {ticker}")
            self.model = load_model(model_path, compile=False)
            self.test_loss = None
            return True

    def generate_forecast(self, stock_data, scaled_data, forecast_days=180):
        if self.model is None:
            st.error("No model available.")
            return None

        last_sequence = scaled_data[-self.time_step:].reshape(self.time_step)
        curr_seq = last_sequence.copy()
        predicted_prices = []

        with st.spinner("Generating forecast..."):
            for _ in range(forecast_days):
                seq_input = curr_seq.reshape(1, self.time_step, 1)
                next_pred = float(self.model.predict(seq_input, verbose=0)[0][0])
                # Add realistic noise based on volatility
                daily_vol = float(self.daily_volatility) if self.daily_volatility is not None else 0.01
                noise_factor = 1 + np.random.normal(0, daily_vol * 1)
                noisy_pred = next_pred * noise_factor
                # Constrain changes reasonably
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
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(base_forecast))
        current_price = float(stock_data['Close'].iloc[-1])
        forecast_price = float(base_forecast[-1])
        potential_return = (forecast_price / current_price - 1) * 100

        # Define risk assessment based on potential return
        if potential_return >= 10:
            risk_assessment = "Low Risk: Strong positive forecast"
        elif potential_return >= 0:
            risk_assessment = "Medium Risk: Moderate gains expected"
        else:
            risk_assessment = "High Risk: Forecast shows negative returns"

        metrics = {
            'ticker': ticker,
            'forecast_date': forecast_dates[-1].strftime('%Y-%m-%d'),
            'current_price': current_price,
            'forecast_price': forecast_price,
            'potential_return': potential_return,
            'risk_assessment': risk_assessment,
            'model_accuracy': f"Test loss: {self.test_loss:.4f}" if self.test_loss is not None else "N/A"
        }
        return metrics

# --- Sidebar Controls ---
canadian_tickers = [
    'AC.TO', 'RY.TO', 'TD.TO', 'ENB.TO', 'SU.TO', 'BMO.TO', 'CP.TO',
    'BNS.TO', 'BCE.TO', 'CM.TO', 'TRP.TO', 'MFC.TO', 'T.TO', 'FTS.TO',
    'QSR.TO', 'SLF.TO', 'CCO.TO', 'L.TO', 'CNQ.TO', 'ATD.TO', 'AQN.TO', 'PPL.TO'
]

st.sidebar.header("Settings")
selected_ticker = st.sidebar.selectbox("Ticker", canadian_tickers)
forecast_horizon = st.sidebar.slider("Forecast Days", min_value=30, max_value=365, value=180, step=30)

# --- Main Flow ---
if st.sidebar.button("Forecast"):
    forecaster = StockForecaster()
    if not os.path.exists(forecaster.base_model_path):
        st.error("Base model not found.")
        st.stop()

    with st.spinner("Loading data..."):
        stock_data = forecaster.download_data(selected_ticker)
        if stock_data is None:
            st.error("Not enough data.")
            st.stop()
        stock_data, scaled_data = forecaster.prepare_data(stock_data)

    # Only retrain if necessary
    if not forecaster.load_or_retrain_model(selected_ticker):
        st.error("Model loading/retraining failed.")
        st.stop()

    base_forecast = forecaster.generate_forecast(stock_data, scaled_data, forecast_days=forecast_horizon)
    if base_forecast is None:
        st.error("Forecast generation failed.")
        st.stop()

    # Create full historical + forecast DataFrame for Plotly
    forecast_dates = pd.date_range(start=stock_data.index[-1] + timedelta(days=1), periods=forecast_horizon)
    df_hist = pd.DataFrame({'Price': stock_data['Close']}, index=stock_data.index)
    df_forecast = pd.DataFrame({'Price': base_forecast}, index=forecast_dates)
    
    # Build Plotly figure showing full data but zoomed to recent history + forecast.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Price'],
                             mode='lines', name="Historical"))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['Price'],
                             mode='lines', name="Forecast", line=dict(color='blue')))
    # Set initial x-axis range: last N days of history plus forecast period.
    first_visible = stock_data.index[-1] - pd.Timedelta(days=forecast_horizon)
    last_visible = forecast_dates[-1]
    fig.update_layout(
        xaxis=dict(range=[first_visible, last_visible]),
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate metrics and show forecast summary
    metrics = forecaster.calculate_forecast_metrics(selected_ticker, stock_data, base_forecast)
    st.subheader("Forecast Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ticker", metrics['ticker'])
        st.metric("Forecast Date", metrics['forecast_date'])
    with col2:
        st.metric("Current Price", f"${metrics['current_price']:.2f}")
        st.metric("Forecast Price", f"${metrics['forecast_price']:.2f}")
    with col3:
        st.metric("Risk Assessment", metrics['risk_assessment'])
        st.metric("Model Accuracy", metrics['model_accuracy'])
else:
    st.info("Select a ticker and click 'Forecast'.")