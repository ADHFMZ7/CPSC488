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


def fetch_price(symbols, start='2009-01-01', end=None):
    out_rows = []
    """
    Downloads historical stock data for a list of symbols (including S&P 500) 
    and returns a DataFrame with columns: date, symbol, open, high, low, close, volume
    """
    # Map for special symbols
    yf_map = {'s&p': '^GSPC'}
    tickers = [yf_map.get(s.lower(), s) for s in symbols]

    print("Downloading data for all symbols")
    print(tickers)
    # all_data = {}

    # for b_ix in range(0, len(tickers), batch_size):
    #     print(f"Downloading batch {b_ix}")
    #     batch = tickers[b_ix:b_ix + batch_size]
    #     df = yf.download(batch, start=start, end=end, group_by='ticker', progress=False, auto_adjust=True, threads=False)
    #     all_data.update({ticker: df[ticker] for ticker in batch if ticker in df.columns.get_level_values(0)})

    # df = pd.concat(all_data, axis=1)
    # print(df)
    
    df = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, auto_adjust=True, threads=False)

    # Looping over each ticker and cleaning its data
    all_rows = []
    for i, s in enumerate(symbols):
        ticker = yf_map.get(s.lower(), s)
        # Extract the DataFrame for each ticker
        if len(tickers) > 1:
            # Multi-column DataFrame
            if ticker in df.columns.levels[1]:
                df_t = df[ticker].copy()
            else:
                print(f"No data found for {s}, skipping.")
                continue
        else:
            # Single ticker DataFrame
            df_t = df.copy()

        df_t = df_t.reset_index()
        df_t['symbol'] = s
        df_t = df_t.rename(columns={'Adj Close':'close', 'Open':'open','High':'high','Low':'low','Volume':'volume','Date':'date'})
        df_t = df_t[['date','symbol','open','high','low','close','volume']]
        all_rows.append(df_t)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=['date','symbol','open','high','low','close','volume'])


# Responsible for downloading the full text of a news article
def fetch_article(url):
    try:
        r = requests.get(url, timeout=10, headers={'User-Agent': 'Assignment-02'})
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join(p.get_text(strip=True) for p in paragraphs)
        return text[:200000]  # limit size to avoid huge CSV
    except Exception:
        return ''


# batch_size controls how many articles are processed at at time 
# max_workers limits the number of threads fetching articles simultaneously (this reduces CPU/memory spikes)
def fetch_articles_in_batches(alln, batch_size=50, max_workers=5): 
    """                                                            
    Fetching articles in batches with limited threads and saving them incrementally
    """
    alln['article'] = '' # create column
    output_file = DATA_DIR/'all_news.csv'

    total = len(alln)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = alln.iloc[start:end].copy()

        # Fetching articles using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_article, url): idx for idx, url in zip(batch.index, batch['URL'])}
            for future in as_completed(futures):
                idx = futures[future]
                alln.at[idx, 'article'] = future.result()

        # Saving the progress incremently
        if start == 0:
            batch.to_csv(output_file, index=False, mode='w')
        else:
            batch.to_csv(output_file, index=False, mode='a', header=False)

        print(f"Fetched articles {start+1} to {end} of {total}")

    return alln
    

def main():
    # Load and merge headlines
    alln = load_headlines()
    alln.to_csv(DATA_DIR/'all_news_raw.csv', index=False)
    
    # Get unique symbols
    symbols = sorted(set(alln['symbol'].dropna().unique()))
    if 's&p' not in symbols:
        symbols.append('s&p')
    
    # Download historical prices
    prices = fetch_price(symbols, start='2009-01-01')
    prices.to_csv(DATA_DIR/'historical_prices.csv', index=False)
    

    # This commented code fetches all of the 3M+ articles 
    # # Fetch full articles for at least 6 years
    # print("Fetching full articles (may take a while)...")
    # alln = fetch_articles_in_batches(alln, batch_size=50, max_workers=5)

    # Fetch full articles in batches for demonstration (first 500 articles)
    print("Fetching full articles in batches (demonstration subset)...")
    demo_alln = alln.head(500).copy()  # limit to first 500 for speed
    demo_alln = fetch_articles_in_batches(demo_alln, batch_size=50, max_workers=5)

    # Replace  original subset in alln with the fetched articles
    alln.loc[demo_alln.index, 'article'] = demo_alln['article']

    # Save final merged dataset
    alln = alln.rename(columns={'URL': 'url'})
    alln = alln[['date','symbol','headline','url','article','publisher']]
    alln.to_csv(DATA_DIR/'all_news.csv', index=False)
    
    print("Saved historical_prices.csv and all_news.csv")

if __name__ == '__main__':
    main()










# headlines = pd.read_csv('../../../data/news_datasets/headlines.csv')
# ratings = pd.read_csv('../../../data/news_datasets/analyst_ratings.csv')

# print(headlines.head())
# print(ratings.head())
