import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("datasets")

def calculate_impact_scores(df):
    """
    Computes returns, volatilities, market adjusted measures and impact scores
    """
    # Sort by symbol / date
    df = df.sort_values(["symbol", "date"]).copy()

    # Daily return
    df["daily_return"] = df.groupby("symbol")["close"].pct_change(fill_method=None)

    # Daily volatility
    df["daily_volatility"] = (
        df.groupby("symbol")["daily_return"]
        .rolling(30)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Compute beta for each stock
    beta_values = {}
    for sym, sub in df.groupby("symbol"):
        if sym == "^GSPC":
            beta_values[sym] = 1.0
            continue

        # Skip if no market_return data available
        if "market_return" not in sub.columns or sub["market_return"].dropna().empty:
            beta_values[sym] = 0
            continue

        if sub["daily_return"].count() > 1:
            # Align lengths
            aligned = sub[["daily_return", "market_return"]].dropna()
            if len(aligned) < 2:
                beta_values[sym] = 0
                continue
            cov = np.cov(aligned["daily_return"], aligned["market_return"])[0][1]
            var = np.var(aligned["market_return"])
            beta_values[sym] = cov / var if var != 0 else 0
        else:
            beta_values[sym] = 0

    # Assign beta
    df["beta"] = df["symbol"].map(beta_values)

    # Alpha, adjusted returns and volatility
    df["alpha"] = df["daily_return"] - df["beta"] * df["market_return"]
    df["market_adj_return"] = df["alpha"]
    df["market_adj_volatility"] = df["daily_volatility"]

    # Normalize
    zr = (df["market_adj_return"] - df["market_adj_return"].mean()) / df["market_adj_return"].std()
    zs = (df["market_adj_volatility"] - df["market_adj_volatility"].mean()) / df["market_adj_volatility"].std()

    # Compute discrete impact score
    impact_score = []
    for r, s in zip(zr, zs):
        if np.isnan(r) or np.isnan(s):
            impact_score.append(np.nan)
        elif abs(r) <= 0.5:
            impact_score.append(0)
        else:
            score = np.sign(r) * (1 + int(abs(r) > 1) + int(s > 1))
            impact_score.append(int(score))

    df["impact_score"] = impact_score
    return df



def main():
    # Load historical prices
    df = pd.read_csv(DATA_DIR / "historical_prices.csv", parse_dates=["date"])

    # Create market_return column using S&P 500 (^GSPC)
    market = (
        df[df["symbol"] == "^GSPC"]
        .sort_values("date")[["date", "close"]]
        .rename(columns={"close": "market_close"})
    )
    market["market_return"] = market["market_close"].pct_change(fill_method=None)

    # Merge market returns into all stocks
    df = df.merge(market[["date", "market_return"]], on="date", how="left")

    # Calculate impact scores
    result_df = calculate_impact_scores(df)

    # Save results
    output_path = DATA_DIR / "historical_prices_impact.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Created {output_path.name}") # Can comment this out


if __name__ == "__main__":
    main()

