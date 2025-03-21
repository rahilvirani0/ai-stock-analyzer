import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="Canadian Stock Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Canadian Stock Market Forecaster")
st.markdown("""
This application uses a two-stage deep learning model to forecast stock prices:
1. A base model trained on the entire Canadian market  
2. A fine-tuned model specific to individual stocks  

The forecast includes confidence intervals to help with investment decisions.
""")

class StockForecaster:
    def __init__(self, time_step=100, base_model_path='canadian_market_base_model.h5'):
        self.time_step = time_step
        self.base_model_path = base_model_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.prediction_errors = None
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
            return None, None, None

        # Calculate volatility
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        self.daily_volatility = stock_data['Daily_Return'].std()

        # Scale data
        scaled_data = self.scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(scaled_data) - self.time_step - 1):
            X.append(scaled_data[i:(i + self.time_step), 0])
            y.append(scaled_data[i + self.time_step, 0])
        if len(X) == 0:
            return None, None, None

        X, y = np.array(X), np.array(y)
        train_size = 0.8
        X_train = X[:int(X.shape[0] * train_size)]
        X_test = X[int(X.shape[0] * train_size):]
        y_train = y[:int(y.shape[0] * train_size)]
        y_test = y[int(y.shape[0] * train_size):]

        # Reshape for LSTM input
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        return (X_train, y_train), (X_test, y_test), scaled_data

    def load_models(self, ticker):
        ticker_model_path = f"{ticker}_fine_tuned_model.h5"
        if os.path.exists(ticker_model_path):
            st.success(f"Loading fine-tuned model for {ticker}")
            self.model = load_model(ticker_model_path, compile=False)
            return True
        elif os.path.exists(self.base_model_path):
            st.warning(f"No fine-tuned model found for {ticker}. Using base market model.")
            self.model = load_model(self.base_model_path, compile=False)
            return True
        else:
            st.error("No models found. Please ensure models are in the current directory.")
            return False

    def analyze_stock(self, ticker):
        if not self.load_models(ticker):
            return None, None, None

        stock_data = self.download_data(ticker)
        if stock_data is None or len(stock_data) < 200:
            st.error(f"Not enough data for {ticker}")
            return None, None, None

        train_test, test_data, scaled_data = self.prepare_data(stock_data)
        if train_test is None or test_data is None:
            st.error(f"Failed to prepare data for {ticker}")
            return None, None, None

        X_test, y_test = test_data
        test_predictions = self.model.predict(X_test)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        test_predictions_actual = self.scaler.inverse_transform(test_predictions)
        self.prediction_errors = y_test_actual - test_predictions_actual

        return stock_data, scaled_data, test_predictions_actual

    def generate_forecast(self, stock_data, scaled_data, forecast_days=180, confidence_width=1.96):
        if self.model is None:
            st.error("No model available. Please ensure models are loaded correctly.")
            return None, None, None
        if stock_data is None or scaled_data is None or len(scaled_data) < self.time_step:
            st.error("Insufficient data for forecasting")
            return None, None, None

        last_sequence = scaled_data[-self.time_step:].reshape(self.time_step)
        if self.prediction_errors is None or len(self.prediction_errors) == 0:
            st.warning("No prediction errors available, using default error value")
            error_std = 0.02
        else:
            error_std = float(np.std(self.prediction_errors))

        last_actual_price = float(self.scaler.inverse_transform([[last_sequence[-1]]])[0][0])
        curr_seq = last_sequence.copy()
        predicted_prices = []
        upper_bound = []
        lower_bound = []

        with st.spinner('Generating forecast...'):
            for _ in range(forecast_days):
                curr_seq_reshaped = curr_seq.reshape(1, self.time_step, 1)
                next_pred = float(self.model.predict(curr_seq_reshaped, verbose=0)[0][0])
                daily_vol = float(self.daily_volatility) if self.daily_volatility is not None else 0.01
                noise_factor = 1 + np.random.normal(0, daily_vol * 0.7)
                noisy_pred = next_pred * noise_factor
                # Constrain prediction changes
                noisy_pred = max(min(noisy_pred, curr_seq[-1] * 1.15), curr_seq[-1] * 0.85)
                predicted_prices.append(noisy_pred)
                upper = noisy_pred + confidence_width * error_std / last_actual_price
                lower = noisy_pred - confidence_width * error_std / last_actual_price
                upper_bound.append(upper)
                lower_bound.append(lower)
                curr_seq = np.append(curr_seq[1:], noisy_pred)

        predicted_array = np.array(predicted_prices).reshape(-1, 1)
        upper_array = np.array(upper_bound).reshape(-1, 1)
        lower_array = np.array(lower_bound).reshape(-1, 1)
        try:
            base_forecast = self.scaler.inverse_transform(predicted_array).flatten()
            upper_forecast = self.scaler.inverse_transform(upper_array).flatten()
            lower_forecast = self.scaler.inverse_transform(lower_array).flatten()
        except Exception as e:
            st.error(f"Error scaling forecast data: {e}")
            return None, None, None

        return base_forecast, upper_forecast, lower_forecast

    def calculate_forecast_metrics(self, ticker, stock_data, base_forecast, upper_forecast, lower_forecast):
        last_date = stock_data.index[-1]
        forecast_days = len(base_forecast)
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

        current_price = float(stock_data['Close'].iloc[-1])
        forecast_price = float(base_forecast[-1])
        upper_price = float(upper_forecast[-1])
        lower_price = float(lower_forecast[-1])

        potential_return_base = (forecast_price / current_price - 1) * 100
        potential_return_upper = (upper_price / current_price - 1) * 100
        potential_return_lower = (lower_price / current_price - 1) * 100

        if potential_return_lower > 10:
            risk_assessment = "Low Risk - Even conservative case shows strong returns"
            risk_color = "green"
        elif potential_return_base > 10 and potential_return_lower > 0:
            risk_assessment = "Medium Risk - Base case shows strong returns, downside is limited"
            risk_color = "blue"
        elif potential_return_base > 0 and potential_return_lower < 0:
            risk_assessment = "Medium-High Risk - Potential for gains but also for losses"
            risk_color = "orange"
        else:
            risk_assessment = "High Risk - Base case and conservative case both negative"
            risk_color = "red"

        model_error = float(np.std(self.prediction_errors))
        mean_price = float(stock_data['Close'].mean())
        relative_error = (model_error / mean_price) * 100

        if relative_error < 5:
            accuracy = "High"
            accuracy_color = "green"
        elif relative_error < 10:
            accuracy = "Medium"
            accuracy_color = "blue"
        else:
            accuracy = "Low"
            accuracy_color = "red"

        metrics = {
            'ticker': ticker,
            'current_price': current_price,
            'forecast_date': forecast_dates[-1].strftime('%Y-%m-%d'),
            'forecast_price': forecast_price,
            'upper_price': upper_price,
            'lower_price': lower_price,
            'potential_return_base': float(potential_return_base),
            'potential_return_upper': float(potential_return_upper),
            'potential_return_lower': float(potential_return_lower),
            'risk_assessment': risk_assessment,
            'risk_color': risk_color,
            'model_accuracy': accuracy,
            'accuracy_color': accuracy_color,
            'relative_error': relative_error
        }
        return metrics


# List of tickers
canadian_tickers = ['AC.TO', 'FM.TO', 'GGD.TO', 'T.TO', 'POU.TO', 'BN-PFG.TO']
other_tickers = []
all_tickers = canadian_tickers + other_tickers

st.sidebar.header("Forecast Settings")
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", options=all_tickers, index=0,
                                         help="Choose a stock to forecast")
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=30, max_value=365,
                                     value=180, step=30, help="Number of days to forecast")
confidence_level = st.sidebar.slider("Confidence Level (%)", min_value=70, max_value=99,
                                     value=95, step=5, help="Confidence level for bounds")
confidence_width = {70: 1.04, 75: 1.15, 80: 1.28, 85: 1.44, 90: 1.64, 95: 1.96, 99: 2.58}[confidence_level]
show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True,
                                      help="Toggle to display or hide the forecast confidence intervals.")

if st.sidebar.button("Generate Forecast"):
    forecaster = StockForecaster()
    if not os.path.exists('canadian_market_base_model.h5'):
        st.error("Base model file 'canadian_market_base_model.h5' not found in the current directory.")
        st.stop()
    with st.spinner(f"Analyzing {selected_ticker}..."):
        stock_data, scaled_data, test_predictions = forecaster.analyze_stock(selected_ticker)
    if stock_data is not None:
        with st.spinner("Generating forecast..."):
            base_forecast, upper_forecast, lower_forecast = forecaster.generate_forecast(
                stock_data, scaled_data, forecast_days=forecast_horizon, confidence_width=confidence_width
            )
        # Prepare interactive DataFrame for the chart
        historical_dates = stock_data.index
        historical_prices = stock_data['Close'].values.flatten()  # flatten to 1D
        forecast_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
        # Create DataFrames for historical and forecast data; include confidence intervals if toggled on
        df_hist = pd.DataFrame({'Historical Prices': historical_prices}, index=historical_dates)
        if show_confidence:
            df_forecast = pd.DataFrame({
                '6-Month Forecast': base_forecast,
                'Upper Bound (95% CI)': upper_forecast,
                'Lower Bound (95% CI)': lower_forecast
            }, index=forecast_dates)
        else:
            df_forecast = pd.DataFrame({
                '6-Month Forecast': base_forecast
            }, index=forecast_dates)
        # Combine the data so that the chart displays both sections interactively
        df_combined = pd.concat([df_hist, df_forecast])
        st.line_chart(df_combined)

        metrics = forecaster.calculate_forecast_metrics(selected_ticker, stock_data, base_forecast, upper_forecast, lower_forecast)
        st.header("Forecast Summary")
        # Adjust metric display based on confidence toggle
        if show_confidence:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Current Price", value=f"${metrics['current_price']:.2f}")
                st.metric(label=f"Forecast Price ({metrics['forecast_date']})", value=f"${metrics['forecast_price']:.2f}",
                          delta=f"{metrics['potential_return_base']:.2f}%")
            with col2:
                st.metric(label="Upper Bound (Optimistic)", value=f"${metrics['upper_price']:.2f}",
                          delta=f"+{metrics['potential_return_upper']:.2f}%")
                st.metric(label="Lower Bound (Conservative)", value=f"${metrics['lower_price']:.2f}",
                          delta=f"{metrics['potential_return_lower']:.2f}%")
            with col3:
                st.markdown(f"""
                **Risk Assessment:**
                <p style='color:{metrics['risk_color']};font-weight:bold;'>{metrics['risk_assessment']}</p>
                
                **Model Accuracy:**
                <p style='color:{metrics['accuracy_color']};font-weight:bold;'>{metrics['model_accuracy']} ({metrics['relative_error']:.2f}% error)</p>
                """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Current Price", value=f"${metrics['current_price']:.2f}")
                st.metric(label=f"Forecast Price ({metrics['forecast_date']})", value=f"${metrics['forecast_price']:.2f}",
                          delta=f"{metrics['potential_return_base']:.2f}%")
            with col2:
                st.markdown(f"""
                **Risk Assessment:**
                <p style='color:{metrics['risk_color']};font-weight:bold;'>{metrics['risk_assessment']}</p>
                
                **Model Accuracy:**
                <p style='color:{metrics['accuracy_color']};font-weight:bold;'>{metrics['model_accuracy']} ({metrics['relative_error']:.2f}% error)</p>
                """, unsafe_allow_html=True)
    else:
        st.error(f"Failed to analyze {selected_ticker}. Please try another stock.")
else:
    st.info("👈 Select a stock and click 'Generate Forecast' to begin analysis")
    st.markdown("""
    ### How to use this application:
    
    1. Select a stock ticker from the dropdown menu  
    2. Adjust the forecast horizon (days to forecast)  
    3. Set your preferred confidence level  
    4. Toggle the display of confidence intervals as needed  
    5. Click "Generate Forecast" to run the analysis
    
    ### About the model:
    
    This application uses a two-stage deep learning approach:
    
    1. **Base Model:** Trained on the entire Canadian stock market to learn general market patterns  
    2. **Fine-tuned Model:** Further trained on specific stocks for more accurate individual predictions
    
    The forecast includes realistic daily fluctuations and confidence intervals to help with investment decisions.
    """)

st.markdown("---")
st.markdown("### Model Information")
st.markdown("""
This application uses:
- A base model trained on multiple Canadian stocks (`canadian_market_base_model.h5`)  
- Fine-tuned models for individual stocks (e.g., `AC.TO_fine_tuned_model.h5`)  
- LSTM neural networks for time series forecasting  
- Historical volatility to create realistic price fluctuations  
- Statistical error analysis for confidence intervals
""")
