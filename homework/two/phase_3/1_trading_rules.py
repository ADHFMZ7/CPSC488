"""
In this file, we prepare the dataset to be used by the 
trading bot in file 2.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path('datasets')


def main():

    # Get the vectorized dataset we decided on
    news = pd.read_parquet(DATA_DIR/'vectorized_news_tfidf.parquet')

    # Get the prices
    prices = pd.read_csv(DATA_DIR/'historical_prices.csv', parse_dates=['date'])
    
    dataset = pd.merge(news, prices, on=['date', 'symbol'], how='inner')
    dataset.to_parquet(DATA_DIR/'bot_dataset.parquet')
    print("Saved bot dataset as bot_dataset.parquet")

if __name__ == "__main__":
    main()
