import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import time
import streamlit as st
import traceback
import math

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.firefox import GeckoDriverManager

from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification

# Initialize the sentiment analysis model for financial news
sentiment_pipe = pipeline(
    "text-classification",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    tokenizer="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    framework="tf"
)
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = TFAutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def create_selenium_driver():
    """
    Create a Selenium WebDriver with robust error handling.
    """
    try:
        # Setup Firefox in headless mode
        firefoxOptions = Options()
        firefoxOptions.add_argument("--headless")
        firefoxOptions.add_argument("--no-sandbox")
        firefoxOptions.add_argument("--disable-dev-shm-usage")
        
        service = Service(GeckoDriverManager().install())
        
        driver = webdriver.Firefox(
            options=firefoxOptions,
            service=service,
        )
        return driver
    except Exception as e:
        st.error(f"Failed to create WebDriver: {e}")
        traceback.print_exc()
        return None

def get_headlines_fallback(ticker):
    """
    Fallback method to fetch headlines using requests if Selenium fails.
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/latest-news/"
    
    try:
        # Add user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            st.error(f"Failed to fetch news. Status code: {response.status_code}")
            return []
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        stream_items = soup.find_all("section", class_="container sz-x-large stream yf-82qtw3 responsive")
        
        def convert_relative_to_date(relative_str):
            # Convert relative time strings to actual dates
            today = datetime.today()

            if "yesterday" in relative_str.lower():
                exact_date = today - timedelta(days=1)
                return exact_date.strftime('%Y-%m-%d')

            # Extract hours from strings like "2 hours ago"
            match = re.search(r'(\d+)\s+hour', relative_str)
            if match:
                hours = int(match.group(1))
                exact_datetime = today - timedelta(hours=hours)
                return exact_datetime.strftime('%Y-%m-%d')

            # Extract minutes from strings like "5 minutes ago"
            match = re.search(r'(\d+)\s+minute', relative_str)
            if match:
                minutes = int(match.group(1))
                exact_datetime = today - timedelta(minutes=minutes)
                return exact_datetime.strftime('%Y-%m-%d')

            # Extract days from strings like "3 days ago"
            match = re.search(r'(\d+)\s+day', relative_str)
            if match:
                days = int(match.group(1))
                exact_date = today - timedelta(days=days)
                return exact_date.strftime('%Y-%m-%d')

            return relative_str  # Already in date format (e.g., "2025-03-25")

        # Extract headlines and dates
        headlines_array = []
        for item in stream_items:
            # Skip ad items
            if "ad-item" in str(item.get('class', '')):
                continue

            # Find headline
            headline_tag = item.find('h3', class_='clamp')
            if not headline_tag:
                continue

            headline = headline_tag.get_text(strip=True)

            # Find date
            pub_div = item.find('div', class_='publishing')
            if pub_div:
                pub_text = pub_div.get_text(strip=True)
                parts = pub_text.split("•")
                relative_time = parts[1].strip() if len(parts) > 1 else pub_text.strip()
                exact_date = convert_relative_to_date(relative_time)
            else:
                exact_date = "No date"

            headlines_array.append({
                "headline": headline,
                "date": exact_date
            })

        return headlines_array
    
    except Exception as e:
        st.error(f"Error in fallback method: {e}")
        traceback.print_exc()
        return []

def get_headlines(ticker):
    """
    Scrape Yahoo Finance latest news page for the given ticker symbol.
    Uses Selenium with a fallback to requests-based method.
    """
    # First, try Selenium method
    driver = create_selenium_driver()
    
    if driver is None:
        st.warning("Selenium WebDriver creation failed. Falling back to alternative method.")
        return get_headlines_fallback(ticker)
    
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/latest-news/"
        
        # Navigate to the page
        driver.get(url)
        
        # Wait and scroll to load more content
        for _ in range(10):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
        
        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        stream_items = soup.find_all("section", class_="container sz-x-large stream yf-82qtw3 responsive")
        
        if not stream_items:
            st.warning("No news items found. Falling back to alternative method.")
            return get_headlines_fallback(ticker)

        def convert_relative_to_date(relative_str):
            # More comprehensive date conversion function
            today = datetime.now()

            # Check for "yesterday"
            if "yesterday" in relative_str.lower():
                exact_date = today - timedelta(days=1)
                return exact_date.strftime('%Y-%m-%d')

            # Check for "hours ago"
            match_hours = re.search(r'(\d+)\s+hour', relative_str)
            if match_hours:
                hours = int(match_hours.group(1))
                exact_date = today - timedelta(hours=hours)
                return exact_date.strftime('%Y-%m-%d')

            # Check for "minutes ago"
            match_minutes = re.search(r'(\d+)\s+minute', relative_str)
            if match_minutes:
                minutes = int(match_minutes.group(1))
                exact_date = today - timedelta(minutes=minutes)
                return exact_date.strftime('%Y-%m-%d')

            # Check for "days ago"
            match_days = re.search(r'(\d+)\s+day', relative_str)
            if match_days:
                days = int(match_days.group(1))
                exact_date = today - timedelta(days=days)
                return exact_date.strftime('%Y-%m-%d')

            # Check for "months ago" (approximate each month as 30 days)
            match_months = re.search(r'(\d+)\s+month', relative_str)
            if match_months:
                months = int(match_months.group(1))
                exact_date = today - timedelta(days=months * 30)
                return exact_date.strftime('%Y-%m-%d')

            # Check for "years ago" (approximate each year as 365 days)
            match_years = re.search(r'(\d+)\s+year', relative_str)
            if match_years:
                years = int(match_years.group(1))
                exact_date = today - timedelta(days=years * 365)
                return exact_date.strftime('%Y-%m-%d')

            # If the relative_str already seems to be a date in YYYY-MM-DD format, return it.
            if re.match(r'\d{4}-\d{2}-\d{2}', relative_str):
                return relative_str

            # Fallback: return today's date if no pattern matched
            return today.strftime('%Y-%m-%d')

        # Process news items
        headlines_array = []
        for item in stream_items:
            # Skip ad items
            if "ad-item" in str(item.get('class', '')):
                continue

            # Find headline
            headline_tag = item.find('h3', class_='clamp')
            if not headline_tag:
                continue

            headline = headline_tag.get_text(strip=True)

            # Find date
            pub_div = item.find('div', class_='publishing')
            if pub_div:
                pub_text = pub_div.get_text(strip=True)
                parts = pub_text.split("•")
                relative_time = parts[1].strip() if len(parts) > 1 else pub_text.strip()
                exact_date = convert_relative_to_date(relative_time)
            else:
                exact_date = "No date"

            headlines_array.append({
                "headline": headline,
                "date": exact_date
            })

        return headlines_array
    
    except WebDriverException as e:
        st.error(f"WebDriver error: {e}")
        traceback.print_exc()
        return get_headlines_fallback(ticker)
    
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        traceback.print_exc()
        return get_headlines_fallback(ticker)
    
    finally:
        # Always close the driver to free up resources
        if driver:
            try:
                driver.quit()
            except:
                pass

def analyze_news(ticker):
    """
    Fetch headlines for a given ticker and perform sentiment analysis.
    Returns a list of dictionaries, each containing:
      - headline
      - date
      - sentiment (label)
      - score (confidence score)
    """
    # Get headlines for the ticker
    headlines = get_headlines(ticker)
    
    # If no headlines found, return empty list
    if not headlines:
        st.warning(f"No news headlines found for ticker {ticker}")
        return []
    
    # Run sentiment analysis on each headline
    results = []
    for item in headlines:
        text = item['headline']
        analysis = sentiment_pipe(text)[0]  # returns dict with 'label' and 'score'
        results.append({
            "headline": text,
            "date": item['date'],
            "sentiment": analysis['label'],
            "score": analysis['score']
        })
    return results

def compute_overall_sentiment(news_results):
    """
    Compute a basic overall sentiment summary from the list of news results.
    Returns a summary string.
    """
    if not news_results:
        return "No news data available."

    # Count occurrences of each sentiment label
    sentiment_counts = {}
    for res in news_results:
        label = res['sentiment']
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

    # Find the most frequent sentiment
    majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return f"Overall News Sentiment: {majority_sentiment} ({sentiment_counts[majority_sentiment]} articles)"

def compute_combined_sentiment(news_results):
    """
    Compute a combined sentiment indicator that integrates:
      1. Weighted Moving Average (WMA) with exponential decay.
      2. Exponential Moving Averages (EMAs) for short and long term.
      3. Sentiment momentum (difference between short-term and long-term EMAs).
    
    Each article's sentiment is mapped to a numeric value:
      - Positive → +score
      - Negative → -score
      - Others (e.g., neutral) → 0

    Returns a dictionary with the following keys:
      - weighted_sentiment
      - ema_short
      - ema_long
      - sentiment_momentum
      - combined_indicator (weighted_sentiment + sentiment_momentum)
    """
    if not news_results:
        return {}

    # Convert date strings to datetime objects
    for res in news_results:
        try:
            res['date_dt'] = datetime.strptime(res['date'], "%Y-%m-%d")
        except Exception:
            res['date_dt'] = datetime.today()  # fallback if conversion fails

    # Sort results by date (oldest first)
    news_results.sort(key=lambda x: x['date_dt'])

    # Map sentiment label to numeric score:
    # Positive becomes +score, Negative becomes -score.
    for res in news_results:
        label = res['sentiment'].lower()
        if "positive" in label:
            res['sentiment_value'] = res['score']
        elif "negative" in label:
            res['sentiment_value'] = -res['score']
        else:
            res['sentiment_value'] = 0

    # Use the most recent date as a reference for decay calculation
    most_recent = max(res['date_dt'] for res in news_results)

    # Weighted Moving Average with exponential decay weight
    lambda_decay = 0.05  # Decay parameter (adjust as needed)
    weighted_sum = 0
    weight_total = 0
    for res in news_results:
        days_diff = (most_recent - res['date_dt']).days
        weight = math.exp(-lambda_decay * days_diff)
        weighted_sum += weight * res['sentiment_value']
        weight_total += weight
    weighted_sentiment = weighted_sum / weight_total if weight_total != 0 else 0

    # Compute two EMAs over time
    ema_short = news_results[0]['sentiment_value']
    ema_long = news_results[0]['sentiment_value']
    alpha_short = 0.3  # Smoothing factor for short-term EMA
    alpha_long = 0.1   # Smoothing factor for long-term EMA
    for res in news_results[1:]:
        ema_short = alpha_short * res['sentiment_value'] + (1 - alpha_short) * ema_short
        ema_long = alpha_long * res['sentiment_value'] + (1 - alpha_long) * ema_long

    # Sentiment Momentum: the difference between the short-term and long-term EMAs
    sentiment_momentum = ema_short - ema_long

    # Combined Indicator (you can adjust this formula as needed)
    combined_indicator = weighted_sentiment + sentiment_momentum

    return {
        "weighted_sentiment": weighted_sentiment,
        "ema_short": ema_short,
        "ema_long": ema_long,
        "sentiment_momentum": sentiment_momentum,
        "combined_indicator": combined_indicator,
    }

# Example Streamlit app
def main():
    st.title("Yahoo Finance News Sentiment Analyzer")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
    
    if st.button("Analyze News Sentiment"):
        with st.spinner("Fetching and analyzing news..."):
            news_results = analyze_news(ticker)
            
            if not news_results:
                st.warning("No news articles found.")
                return
            
            st.write(f"## News Headlines for {ticker}")
            for result in news_results:
                st.write(f"**{result['headline']}**")
                st.write(f"Date: {result['date']}")
                st.write(f"Sentiment: {result['sentiment']} (Confidence: {result['score']:.2f})")
                st.write("---")
            
            overall_sentiment = compute_overall_sentiment(news_results)
            st.write(f"## {overall_sentiment}")

            # Compute and display combined sentiment metrics
            combined_metrics = compute_combined_sentiment(news_results)
            if combined_metrics:
                st.write("## Combined Sentiment Metrics")
                st.write(f"**Weighted Sentiment:** {combined_metrics['weighted_sentiment']:.4f}")
                st.write(f"**Short-Term EMA:** {combined_metrics['ema_short']:.4f}")
                st.write(f"**Long-Term EMA:** {combined_metrics['ema_long']:.4f}")
                st.write(f"**Sentiment Momentum:** {combined_metrics['sentiment_momentum']:.4f}")
                st.write(f"**Combined Indicator:** {combined_metrics['combined_indicator']:.4f}")
            else:
                st.write("No combined sentiment metrics available.")

if __name__ == "__main__":
    main()