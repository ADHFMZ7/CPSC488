import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("datasets")

def calculate_impact_scores(df):
    """
    Computes returns, volatilities, market adjusted measures and impact scores
    """
    # Sort by symbol / date
    df = df.sort_values(['symbol', 'date'])

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
    df = pd.read_csv(DATA_DIR / "volatility.csv", parse_dates=["date"])

    # Calculate impact scores
    result_df = calculate_impact_scores(df)

    # Save results
    output_path = DATA_DIR / "historical_prices_impact.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Created {output_path.name}")


if __name__ == "__main__":
    main()

