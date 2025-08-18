import requests
import argparse
import json

def fetch_alpha_vantage_news(ticker: str, api_key: str):
    """
    Fetches news and sentiment from Alpha Vantage for a given ticker.
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract headlines and sentiment
        headlines = [item['title'] for item in data.get('feed', [])]
        
        # Calculate an overall sentiment score
        # The API provides sentiment per article, so we'll average them.
        sentiment_score = 0
        sentiment_count = 0
        for item in data.get('feed', []):
            for sentiment in item.get('ticker_sentiment', []):
                if sentiment['ticker'] == ticker:
                    try:
                        sentiment_score += float(sentiment['ticker_sentiment_score'])
                        sentiment_count += 1
                    except (ValueError, TypeError):
                        continue # Ignore if score is not a valid float
        
        overall_sentiment = sentiment_score / sentiment_count if sentiment_count > 0 else 0
        
        return headlines, overall_sentiment

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from Alpha Vantage: {e}")
        return [], 0
    except json.JSONDecodeError:
        print("Error decoding JSON response from Alpha Vantage.")
        return [], 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch news headlines and sentiment from Alpha Vantage.")
    parser.add_argument("ticker", type=str, help="The stock ticker to fetch news for.")
    parser.add_argument("api_key", type=str, help="Your Alpha Vantage API key.")
    args = parser.parse_args()

    headlines, sentiment = fetch_alpha_vantage_news(args.ticker, args.api_key)
    
    output = {
        "headlines": headlines,
        "sentiment": sentiment
    }
    
    print(json.dumps(output))