import time
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from newspaper import Article
import trafilatura


DATA_DIR = Path('datasets')
DATA_DIR.mkdir(exist_ok=True)


def load_headlines():
    # Load datasets
    a = pd.read_csv(DATA_DIR/'analyst_ratings.csv', parse_dates=['date'])
    b = pd.read_csv(DATA_DIR/'headlines.csv', parse_dates=['date'])
    
    # Standardize column names
    a = a.rename(columns={'stock':'symbol', 'url':'URL'})
    b = b.rename(columns={'stock':'symbol', 'url':'URL'})
    
    # Merge datasets
    alln = pd.concat([a, b], ignore_index=True, sort=False)

    alln['date'] = alln.date.astype(str).apply(lambda x: x.strip().split(' ')[0])
    alln['date'] = pd.to_datetime(alln['date'], format='%Y-%m-%d')
    
    # Keep only required columns
    alln = alln[['date', 'symbol', 'headline', 'URL', 'publisher']]
    
    return alln


def fetch_price(symbols, start='2010-01-01', end='2016-12-31', batch_size=128):
    """
    Downloads historical stock data for a list of symbols (including S&P 500) 
    and returns a DataFrame with columns: date, symbol, open, high, low, close, volume
    """

    tickers = list(set(symbols)) # list({yf_map.get(s.lower(), s) for s in symbols})
    tickers.append('^GSPC')

    print(f"Downloading data from {len(tickers)} tickers")

    all_data = pd.DataFrame()

    for b_ix in range(0, len(tickers), batch_size):
        
        print(f"Downloading batch {b_ix//batch_size}/{len(tickers)//batch_size}:")
        batch = tickers[b_ix:b_ix + batch_size]

        df = yf.download(batch, start=start, end=end, group_by='ticker', auto_adjust=False, threads=8, keepna=False)

        if df.empty:
            continue

        prices = df.stack(level=0, future_stack=True).reset_index().rename(columns={
            'Date':'date',
            'Open':'open',
            'High':'high',
            'Low':'low',
            'Adj Close':'close',
            'Volume':'volume',
            'Ticker':'symbol'
        })[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]


        all_data = pd.concat([all_data, prices])
        time.sleep(2)
   
    all_data['symbol'] = all_data['symbol'].replace('^GSPC', 's&p')
    return all_data


def extract_article(url):
    try:    
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
        text = trafilatura.extract(html)
        if not text:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
        return text
    except Exception as e:
        return None

def fetch_articles(df, start='2010-01-01', end='2016-12-31', n=5000):

    # df of form Index(['date', 'symbol', 'headline', 'URL', 'publisher'], dtype='object')

    # subset only in 6 year span
    mask = (df['date'] >= start) & (df['date'] <= end)
    df = df.loc[mask]

    # Get 25 most frequent tickers
    # top_symbols = df.symbol.value_counts().head(25).index.tolist()
    # print(top_symbols)
    # subset = df[df['symbol'].isin(top_symbols)]#.sample(n, random_state=42)
    subset = df.sample(n, random_state=42)
    print(subset)
    tqdm.pandas(desc="Fetching article")
    subset['article'] = subset.URL.progress_apply(extract_article)

    print(subset)
    return subset


def main():

    # Load and merge headlines
    alln = load_headlines()
    alln.to_csv(DATA_DIR/'all_news_raw.csv', index=False)

    # tickers = sorted(set(alln['symbol'].dropna().unique()))
    # print(tickers)

    # Download historical prices
    # prices = fetch_price(tickers, batch_size=64)
    # print(prices.head())
    # prices.to_csv(DATA_DIR/'historical_prices.csv', index=False)


    prices = pd.read_csv('datasets/historical_prices.csv')

    # remove articles without symbol in prices
    alln = alln[alln['symbol'].isin(prices['symbol'].unique())]
    alln.reset_index(drop=True)

    # print(prices.head())


    # This commented code fetches all of the 3M+ articles 
    # Fetch full articles for at least 6 years
    # print("Fetching full articles (may take a while)...")
    alln = fetch_articles(alln)

    # Fetch full articles in batches for demonstration (first 500 articles)
    # print("Fetching full articles in batches (demonstration subset)...")
    # demo_alln = alln.head(500).copy()  # limit to first 500 for speed
    # demo_alln = fetch_articles_in_batches(demo_alln, batch_size=50, max_workers=5)

    # Replace  original subset in alln with the fetched articles
    # alln.loc[demo_alln.index, 'article'] = demo_alln['article']

    # Save final merged dataset
    alln = alln[['date','symbol','headline','URL','article','publisher']]
    alln.to_csv(DATA_DIR/'all_news.csv', index=False)
    
    print("Saved historical_prices.csv and all_news.csv")

if __name__ == '__main__':
    main()

