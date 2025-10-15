import pandas as pd
import yfinance as yf
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup


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

    print("Downloading data for all symbols at once...")
    # Download all tickers at once
    df = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, auto_adjust=True)

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
    

def fetch_article(url):
    try:
        r = requests.get(url, timeout=10, headers={'User-Agent': 'Assignment-02'})
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join(p.get_text(strip=True) for p in paragraphs)
        return text[:200000]  # limit size to avoid huge CSV
    except Exception:
        return ''
    

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
    
    # Fetch full articles for at least 6 years
    print("Fetching full articles (may take a while)...")
    alln['article'] = alln['URL'].apply(lambda u: fetch_article(u) if pd.notnull(u) else '')
    
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


