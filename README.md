# AI Trading Assistant: Real-time Sentiment & Forecasting Engine

## Overview

This project combines machine learning stock price prediction with news sentiment analysis to create a comprehensive trading assistant that provides forecasts with sentiment context. The system uses LSTM neural networks for price prediction and a fine-tuned DistilRoBERTa model for sentiment analysis of financial news.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Development Process](#development-process)
- [Technical Implementation](#technical-implementation)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Features

- **Stock Price Forecasting**: LSTM-based models that can predict stock prices up to 365 days into the future
- **Real-time News Sentiment Analysis**: Analyzes latest news headlines for any given stock
- **Combined Analysis**: Integrates price forecasts with sentiment indicators
- **Model Fine-tuning**: Pre-trained base model that gets fine-tuned for individual stocks
- **Backtesting**: Includes model evaluation through backtesting with RMSE and MAPE metrics
- **Interactive Visualization**: Plotly-based charts showing both price forecasts and sentiment trends
- **Risk Assessment**: Provides risk evaluation based on forecast potential returns

## Architecture

```
├── streamlit_app.py                # Main web application
├── retrain_stock_model.py          # Stock model fine-tuning module
├── backtest_model.py               # Accuracy evaluation module
├── news_sentiment.py               # News scraping and sentiment analysis
├── canadian_market_base_model.h5   # Base model trained on Canadian market
└── stock-models/                   # Directory for stored models
    └── {TICKER}_fine_tuned_model_{DATE}.h5  # Stock-specific models
```

## Development Process

My development process followed these main stages:

### 1. LSTM Model Development
I started by exploring various LSTM architectures through iterative experimentation in Google Colab. After reviewing existing implementations, I modified and enhanced the models to achieve better forecasting performance.

The model development followed multiple iterations:
- Initial architecture testing with different layer configurations
- Hyperparameter tuning (learning rate, epochs, batch size)
- Sequence length optimization
- Loss function comparison (MSE vs. MAE)
- Adding noise to make predictions more realistic

### 2. News Sentiment Integration
To incorporate market sentiment, I implemented:
- A web scraper for Yahoo Finance news using both Selenium and a fallback method
- Integration with the HuggingFace DistilRoBERTa model fine-tuned for financial sentiment
- Development of time-weighted sentiment metrics to prioritize recent news
- Creation of exponential moving averages to detect sentiment momentum

### 3. Streamlit Application Design
The final stage involved:
- Building an interactive web interface with Streamlit
- Creating visualizations that combine price and sentiment data
- Implementing model retraining and caching for performance
- Adding risk assessment metrics

## Technical Implementation

### LSTM Model for Stock Prediction

The core of the price forecasting engine is a deep learning model using LSTM (Long Short-Term Memory) architecture. The model:

1. Takes a sequence of historical prices (default: 100 days)
2. Processes this through LSTM layers to capture temporal patterns
3. Outputs a single prediction for the next day's price

The unique aspect of my implementation is the two-stage training approach:
- First, a base model is trained on 100+ Canadian stocks to capture general market patterns
- Then, this base model is fine-tuned for specific stocks to capture individual stock behavior

This transfer learning approach significantly improves performance, especially for stocks with limited historical data.

### News Sentiment Analysis

The sentiment analysis module:

1. Scrapes recent news headlines from Yahoo Finance
2. Processes each headline through a DistilRoBERTa model fine-tuned for financial sentiment
3. Calculates multiple sentiment metrics:
   - Weighted sentiment with exponential decay to prioritize recent news
   - Short-term (3-day) and long-term (10-day) exponential moving averages
   - Momentum indicator showing sentiment direction

The combined indicator integrates both current sentiment level and directional momentum.

## Key Components

### `retrain_stock_model.py`

This module handles the fine-tuning of stock-specific models:

```python
def retrain_stock_model(base_model_path, target_ticker, epochs=8, time_step=100, model_folder='stock-models'):
```

Key features:
- Downloads recent stock data for the target ticker
- Prepares data sequences for LSTM training
- Loads the base model and fine-tunes it on target stock data
- Manages model versioning by date
- Cleans up older models for the same stock

### `backtest_model.py`

Evaluates model accuracy using historical data:

```python
def backtest_model_accuracy(model, scaler, stock_data, time_step=100, forecast_horizon=30):
```

The backtest uses a sliding window approach to simulate real trading conditions.

### `news_sentiment.py`

Handles news scraping and sentiment analysis:

```python
def analyze_news(ticker):
```

```python
def compute_combined_sentiment(news_results):
```

Key features:
- Robust scraping with Selenium and fallback methods
- Integration with HuggingFace transformers
- Advanced sentiment metrics calculation
- Time-weighting to prioritize recent news

### `streamlit_app.py`

The main application ties everything together:

```python
class StockForecaster:
```

The `StockForecaster` class manages:
- Data downloading and preparation
- Model loading/retraining
- Forecast generation with realistic noise
- Metrics calculation

The UI components display:
- Interactive price charts with forecast
- Sentiment markers on historical prices
- Sentiment metrics dashboard
- Risk assessment summary

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the base model or train your own
4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

1. Select a Canadian stock ticker from the dropdown
2. Adjust the forecast horizon (30-365 days)
3. Click "Forecast" to generate predictions
4. Analyze the combined price and sentiment chart
5. Review the risk assessment and model accuracy

## Future Improvements

- Add macroeconomic indicators (interest rates, GDP, etc.)
- Implement ensemble models for more robust forecasting
- Add portfolio optimization based on forecasts
- Create alert system for significant sentiment shifts
- Expand to international markets beyond Canadian stocks

---

This project represents my exploration of combining traditional time-series forecasting with sentiment analysis to create a more holistic market prediction tool. While no model can perfectly predict the market, this approach provides valuable context by incorporating both price patterns and market sentiment.