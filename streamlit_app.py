import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the retraining function and backtesting function
from retrain_stock_model import retrain_stock_model
from backtest_model import backtest_model_accuracy
# Import news sentiment functions
from news_sentiment import analyze_news, compute_overall_sentiment

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
        self.test_loss = None

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
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        self.daily_volatility = stock_data['Daily_Return'].std()
        scaled_data = self.scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
        return stock_data, scaled_data

    def load_or_retrain_model(self, ticker):
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
                daily_vol = float(self.daily_volatility) if self.daily_volatility is not None else 0.01
                noise_factor = 1 + np.random.normal(0, daily_vol * 0.7)
                noisy_pred = next_pred * noise_factor
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
        if potential_return >= 10:
            risk_assessment = "Low Risk: Strong positive"
        elif potential_return >= 0:
            risk_assessment = "Medium Risk: Moderate gains"
        else:
            risk_assessment = "High Risk: Negative returns"
        metrics = {
            'ticker': ticker,
            'forecast_date': forecast_dates[-1].strftime('%Y-%m-%d'),
            'current_price': current_price,
            'forecast_price': forecast_price,
            'potential_return': potential_return,
            'risk_assessment': risk_assessment
        }
        return metrics

# --- Sidebar ---
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
    
    if not forecaster.load_or_retrain_model(selected_ticker):
        st.error("Model loading/retraining failed.")
        st.stop()
    
    base_forecast = forecaster.generate_forecast(stock_data, scaled_data, forecast_days=forecast_horizon)
    if base_forecast is None:
        st.error("Forecast generation failed.")
        st.stop()
    
    # Backtest model accuracy using a 30-day backtest window
    rmse, mape = backtest_model_accuracy(
        forecaster.model,
        forecaster.scaler,
        stock_data,
        time_step=forecaster.time_step,
        forecast_horizon=30
    )
    
    # Prepare data for the interactive chart: plot full historical data and forecast
    last_date = stock_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
    df_hist = pd.DataFrame({'Price': stock_data['Close'].values.flatten()}, index=stock_data.index)
    df_forecast = pd.DataFrame({'Price': base_forecast}, index=forecast_dates)
    forecastColor = 'blue'
    metrics = forecaster.calculate_forecast_metrics(selected_ticker, stock_data, base_forecast)
    if metrics['forecast_price'] > metrics['current_price']:
        forecastColor = '#39FF14'
    elif metrics['current_price'] > metrics['forecast_price']:
        forecastColor = '#ff1818'
    
    # Create a Plotly figure with secondary y-axis for sentiment
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add historical and forecast price traces on primary y-axis
    fig.add_trace(go.Scatter(
        x=df_hist.index, y=df_hist['Price'],
        mode='lines', name="Historical",
        line=dict(color='#6a6a6a')
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=df_forecast.index, y=df_forecast['Price'],
        mode='lines', name="Forecast",
        line=dict(color=forecastColor)
    ), secondary_y=False)
    
    # --- Add News Sentiment Markers on Historical Data ---
    news_results = analyze_news(selected_ticker)
    news_markers = []
    for news in news_results:
        try:
            news_date = datetime.strptime(news["date"], "%Y-%m-%d")
        except Exception:
            continue
        if news_date < stock_data.index[0] or news_date > stock_data.index[-1]:
            continue
        nearest_date = min(stock_data.index, key=lambda d: abs(d - news_date))
        price_at_date = stock_data.loc[nearest_date, "Close"]
        news_markers.append({"date": nearest_date, "price": price_at_date, "news": news})
    
    from collections import defaultdict
    grouped_markers = defaultdict(list)
    for marker in news_markers:
        grouped_markers[marker["date"]].append(marker)
    
    price_range = stock_data["Close"].max() - stock_data["Close"].min()
    offset_step = price_range * 0.01
    for date, markers in grouped_markers.items():
        n = len(markers)
        for i, marker in enumerate(markers):
            offset = offset_step * (i - (n - 1) / 2)
            x_val = marker["date"]
            y_val = marker["price"] + offset
            sentiment = marker["news"]["sentiment"].lower()
            if sentiment == "positive":
                symbol = "triangle-up"
                color = "#00cc00"
            elif sentiment == "negative":
                symbol = "triangle-down"
                color = "#ff0000"
            else:  # neutral
                symbol = "diamond"
                color = "#808080"
            fig.add_trace(go.Scatter(
                x=[x_val],
                y=[y_val],
                mode="markers",
                marker_symbol=symbol,
                marker=dict(color=color, size=16, opacity=0.9),
                showlegend=False,
                hovertemplate=(
                    f"Headline: {marker['news']['headline']}<br>"
                    f"Date: {marker['news']['date']}<br>"
                    f"Sentiment: {marker['news']['sentiment']}<br>"
                    f"Confidence: {marker['news']['score']:.2f}<extra></extra>"
                )
            ), secondary_y=False)
    
    # --- Advanced News Sentiment Analysis ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("News Sentiment Analysis")
    sentiment_forecast = None
    if news_results:
        # Convert news_results to a DataFrame for processing
        news_df = pd.DataFrame(news_results)
        news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
        news_df = news_df.dropna(subset=['date'])
        
        # Map sentiment text to a numeric value: positive=1, neutral=0, negative=-1
        sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
        news_df['numeric_sentiment'] = news_df['sentiment'].str.lower().map(sentiment_mapping)
        
        # Sort by date and compute days difference from the latest date
        news_df = news_df.sort_values('date')
        max_date = news_df['date'].max()
        news_df['days_diff'] = (max_date - news_df['date']).dt.days
        
        # Compute exponentially decaying weights (Weighted Sentiment)
        decay_lambda = 0.1  # adjust decay rate as needed
        news_df['weight'] = np.exp(-decay_lambda * news_df['days_diff'])
        weighted_sentiment = (news_df['numeric_sentiment'] * news_df['weight']).sum() / news_df['weight'].sum()
        
        # Group by date (daily average sentiment)
        daily_sentiment = news_df.groupby(news_df['date'].dt.date)['numeric_sentiment'].mean().reset_index()
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        daily_sentiment = daily_sentiment.sort_values('date')
        
        # Calculate short-term and long-term EMAs on daily sentiment
        short_span = 3
        long_span = 10
        daily_sentiment['short_ema'] = daily_sentiment['numeric_sentiment'].ewm(span=short_span, adjust=False).mean()
        daily_sentiment['long_ema'] = daily_sentiment['numeric_sentiment'].ewm(span=long_span, adjust=False).mean()
        daily_sentiment['momentum'] = daily_sentiment['short_ema'] - daily_sentiment['long_ema']
        
        latest_momentum = daily_sentiment.iloc[-1]['momentum']
        # Combined Indicator: sum of weighted sentiment and latest momentum
        combined_indicator = weighted_sentiment + latest_momentum
        
        # Compute a sentiment forecast over the forecast horizon.
        # Here we assume a simple linear forecast:
        sentiment_forecast = combined_indicator + np.linspace(0, latest_momentum, forecast_horizon)
        
        # Display the advanced news sentiment metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Weighted Sentiment", f"{weighted_sentiment:.3f}")
        with col2:
            st.metric("Short EMA", f"{daily_sentiment.iloc[-1]['short_ema']:.3f}")
        with col3:
            st.metric("Long EMA", f"{daily_sentiment.iloc[-1]['long_ema']:.3f}")
        with col4:
            st.metric("Momentum", f"{latest_momentum:.3f}")
        with col5:
            st.metric("Combined Indicator", f"{combined_indicator:.3f}")
        
        overall_sentiment = compute_overall_sentiment(news_results)
        st.markdown(f"**Overall News Sentiment:** {overall_sentiment}")
    
    # --- Overlay Sentiment Forecast on Graph ---
    # If sentiment forecast is available, add it to the figure on the secondary y-axis.
    if sentiment_forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=sentiment_forecast,
            mode='lines', name="Sentiment Forecast",
            line=dict(color='purple', dash='dash')
        ), secondary_y=True)
        # Set a fixed range for sentiment axis (adjust as needed)
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment", secondary_y=True, range=[-1.5, 1.5])
    
    first_visible = stock_data.index[-1] - pd.Timedelta(days=forecast_horizon)
    last_visible = forecast_dates[-1]
    fig.update_layout(
        xaxis=dict(range=[first_visible, last_visible]),
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Forecast Summary Section ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Forecast Summary")
    col1, col2, col3 = st.columns([0.5, 0.5, 1])
    with col1:
        st.metric("Ticker", metrics['ticker'])
        st.metric("Forecast Date", metrics['forecast_date'])
    with col2:
        st.metric("Current Price", f"${metrics['current_price']:.2f}")
        st.metric("Forecast Price", f"${metrics['forecast_price']:.2f}")
    with col3:
        st.metric("Risk Assessment", metrics['risk_assessment'])
        st.metric("Model Accuracy", f"RMSE: ${rmse:.2f}, MAPE: {mape:.2f}%")
    
    # --- Detailed News Section ---
    if news_results:
        st.subheader("News Details")
        if 'news_df' in locals():
            st.dataframe(news_df)
        else:
            st.dataframe(pd.DataFrame(news_results))
    else:
        st.info("No news data available for this ticker.")
else:
    st.info("Select a ticker and click 'Forecast'.")