import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("datasets")

def compute_volatility(df):
    """
    Computes dail returns and rolling volatility for each ticker
    """

    df = df.sort_values(['symbol', 'date']).copy()
    df['daily_return'] = df.groupby('symbol')['close'].pct_change(fill_method=None)
    df['volatility_30d'] = df.groupby('symbol')['daily_return'].rolling(30).std().reset_index(level=0, drop=True) # The higher the std, the more volatile 
    return df

def main():
    # Loading historical prices
    price_df = pd.read_csv(DATA_DIR / 'historical_prices.csv', parse_dates=['date'])

    # Computing volatility
    vol_df = compute_volatility(price_df)

    # Saveing the results to csv file
    vol_df.to_csv(DATA_DIR / 'volatility.csv', index=False)
    print("Volatility data saved to volatility.csv")

if __name__ == "__main__":
    main()
