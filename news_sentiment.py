# news_sentiment.py
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification

# Initialize the sentiment analysis pipeline using TensorFlow classes
sentiment_pipe = pipeline(
    "text-classification",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    tokenizer="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    framework="tf"
)
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = TFAutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def get_headlines(ticker):
    """
    Scrape Yahoo Finance news page for the given ticker symbol.
    Returns a list of dictionaries with keys 'headline' and 'date'.
    """
    url = f"https://ca.finance.yahoo.com/quote/{ticker}/news/"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/112.0.0.0 Safari/537.36")
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    stream_items = soup.find("ul", class_="stream-items")
    if not stream_items:
        return []

    def convert_relative_to_date(relative_str):
        today = datetime.today()
        match = re.search(r'(\d+)\s+day', relative_str)
        if match:
            days = int(match.group(1))
            exact_date = today - timedelta(days=days)
            return exact_date.strftime('%Y-%m-%d')
        return relative_str

    headlines_array = []
    for item in stream_items.find_all("li"):
        classes = item.get("class", [])
        if "story-item" in classes and "ad-item" not in classes:
            headline_tag = item.find("h3")
            headline = headline_tag.get_text(strip=True) if headline_tag else "No headline"
            date_div = item.find("div", class_="publishing")
            if date_div:
                pub_text = date_div.get_text(strip=True)
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

def analyze_news(ticker):
    """
    Fetch headlines for a given ticker and perform sentiment analysis.
    Returns a list of dictionaries, each containing:
      - headline
      - date
      - sentiment (label)
      - score (confidence score)
    """
    headlines = get_headlines(ticker)
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
    Compute overall sentiment summary from the list of news results.
    Returns a summary string.
    """
    if not news_results:
        return "No news data available."

    sentiment_counts = {}
    for res in news_results:
        label = res['sentiment']
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

    majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return f"Overall News Sentiment: {majority_sentiment} ({sentiment_counts[majority_sentiment]} articles)"