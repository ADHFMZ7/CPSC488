import pandas as pd
import yfinance as yf
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    
    # Keep only required columns
    alln = alln[['date', 'symbol', 'headline', 'URL', 'publisher']]
    
    return alln


def fetch_price(symbols, start='2010-01-01', end='2016-12-31', batch_size=128):
    out_rows = []
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

    return all_data


def fetch_articles(urls, max_workers=10, timeout=10):
    """
    Fetches full article text from a list of urls
    """
    
    def fetch_single(url):
        try:
            response = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code != 200:
                return url, None
            
            soup = BeautifulSoup(response.text, 'html.parser')

            # Attempting to extract artcle text from common HTML tags
            paragraphs = soup.find_all('p')
            article_text = ' '.join(p.get_text(strip=True) for p in paragraphs)

            # Cleanup
            article_text = article_text.replace('\xa0', ' ').strip()

            if len(article_text) < 100:
                return url, None
            
            return url, article_text
        
        except Exception as e:
            print(f"Skipped: {url}")
            return url, None
    print(f"Fetching {len(urls)} articles...")
    results = []

    # Parallel downloading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_single, url): url for url in urls}
        for future in as_completed(future_to_url):
            url, article = future.result()
            results.append({'URL': url, 'article': article})


            df = pd.DataFrame(results)
            return df


def main():

    # Load and merge headlines
    alln = load_headlines()
    alln.to_csv(DATA_DIR/'all_news_raw.csv', index=False)

    tickers = sorted(set(alln['symbol'].dropna().unique()))
    # print(tickers)

    # Download historical prices
    prices = fetch_price(tickers, batch_size=64)
    print(prices.head())
    prices.to_csv(DATA_DIR/'historical_prices.csv', index=False)

    # This commented code fetches all of the 3M+ articles 
    # Fetch full articles for at least 6 years
    # print("Fetching full articles (may take a while)...")
    # alln = fetch_articles(alln5)

    # Fetch full articles in batches for demonstration (first 500 articles)
    # print("Fetching full articles in batches (demonstration subset)...")
    # demo_alln = alln.head(500).copy()  # limit to first 500 for speed
    # demo_alln = fetch_articles_in_batches(demo_alln, batch_size=50, max_workers=5)

    # Replace  original subset in alln with the fetched articles
    # alln.loc[demo_alln.index, 'article'] = demo_alln['article']

    # Save final merged dataset
    # alln = alln.rename(columns={'URL': 'url'})
    # alln = alln[['date','symbol','headline','url','article','publisher']]
    # alln.to_csv(DATA_DIR/'all_news.csv', index=False)
    
    print("Saved historical_prices.csv and all_news.csv")

    # Fetching full article
    demo_alln = alln.head(500).copy()
    article_df = fetch_articles(demo_alln['URL'])
    demo_alln = demo_alln.merge(article_df, on='URL', how='left')


    # Save combined dataset
    demo_alln.to_csv(DATA_DIR / 'all_news.csv', index=False)


if __name__ == '__main__':
    main()


