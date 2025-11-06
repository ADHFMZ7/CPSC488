import pandas as pd
from pathlib import Path

DATA_DIR = Path('datasets')

def split_dataset(df, split=0.8):
    df.sort_values("date") 
    n_train = int(len(df) * split)

    return df.iloc[:n_train], df.iloc[n_train:]


def main():


    for name in ["dtm", "tfidf", "curated"]:

        df = pd.read_parquet(DATA_DIR/f"vectorized_news_{name}.parquet")

        df = df.dropna(subset=['news_vector', 'impact_score'])

        train_df, test_df = split_dataset(df)
        train_df.to_parquet(DATA_DIR/f"train_{name}.parquet")
        test_df.to_parquet(DATA_DIR/f"test_{name}.parquet")


if __name__ == '__main__':
    main()
