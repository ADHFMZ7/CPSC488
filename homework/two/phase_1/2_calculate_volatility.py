import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("datasets")

def compute_volatility(price_df):

    # Daily return
    price_df["daily_return"] = np.log(1 + price_df.groupby('symbol')['close'].pct_change(periods=3))


    # Daily volatility: The standard deviation of the last 3 days' return
    price_df['daily_volatility'] = (price_df.groupby('symbol').daily_return
                                           .rolling(window=3)
                                           .std()
                                           .reset_index(level=0, drop=True)
    )


    # Market-adj return and volatility
    market = price_df[price_df.symbol == 's&p'][['date', 'daily_return']]
    market = market.rename(columns={'daily_return': 'market_return'})
    price_df = price_df.merge(market, on='date', how='left')

    price_df = price_df.dropna(subset=['daily_return', 'market_return'])

    group_stats = price_df.groupby('symbol').agg(

        mean_stock=('daily_return', 'mean'),
        mean_market=('market_return', 'mean'),
        cov=('daily_return', lambda x: np.cov(x, price_df.loc[x.index, 'market_return'])[0, 1]),
        var_market=('market_return', lambda x: np.var(price_df.loc[x.index, 'market_return'], ddof=1))
    )

    group_stats['beta'] = group_stats['cov'] / group_stats['var_market']
    group_stats['alpha'] = group_stats['mean_stock'] - group_stats['beta'] * group_stats['mean_market']

    price_df = price_df.merge(group_stats[['alpha', 'beta']], on='symbol',how='left')  

    price_df['idiosyn_return'] = price_df.daily_return - (price_df.alpha + price_df.beta * price_df.market_return)
    
    idiosyn_vol = price_df.groupby('symbol')['idiosyn_return'].std()
    price_df = price_df.merge(idiosyn_vol.rename('idiosyn_volatility'), on='symbol', how='left')

    price_df['market_adj_return'] = price_df.alpha + price_df.beta * price_df.market_return 
    price_df['market_adj_volatility'] = None



def main():
    # Loading historical prices
    price_df = pd.read_csv(DATA_DIR / 'historical_prices.csv', parse_dates=['date'])
    price_df = price_df.replace(to_replace='^GSPC', value='s&p')
    price_df = price_df.sort_values(['symbol', 'date']).copy()

    price_df = compute_volatility(price_df)

    # Saveing the results to csv file
    price_df.to_csv(DATA_DIR / 'volatility.csv', index=False)
    print("Volatility data saved to volatility.csv")

if __name__ == "__main__":
    main()
