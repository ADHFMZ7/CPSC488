import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK stopwords and tokenizer are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = stopwords.words("english")
DATA_DIR = Path("datasets") 

# Defining top sentiment words
SENTIMENT_WORDS = [
    "gain", "loss", "strong", "weak", "win", "downgrade", "rise", "fall", "buy", "sell"
]

# Text proecssing, cleaning and tokenizing text for sentiment analysis
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctiation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    # Join back to string
    return " ".join(tokens)

# Aggregate news over three consecutive trading days
def aggregate_news(df):
    df= df.sort_values(["symbol", "date"]).copy()
    df["news"] = df["headline"].fillna("") + " " + df["article"].fillna("")

    aggregated_rows = []
    for symbol, group in df.groupby("symbol"):
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            start = max(0, i - 2) 
            news_window = " ".join(group.loc[start:i, "news"])
            aggregated_rows.append({
                "date": group.loc[i, "date"],
                "symbol": symbol,
                "news": news_window
            })
    return pd.DataFrame(aggregated_rows)

# Vectorization methods
def vectorize_news(df, method="dtm"):
    preprocessed_news = df["news"].apply(preprocess_text)

    if method == "dtm":
        vec = CountVectorizer(stop_words="english")
        matrix = vec.fit_transform(preprocessed_news)
    elif method == "tfidf":
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(preprocessed_news)
    elif method == "curated":
        matrix = np.zeros((len(df), len(SENTIMENT_WORDS)), dtype=int)
        for i, text in enumerate(preprocessed_news):
            for j, word in enumerate (SENTIMENT_WORDS):
                matrix[i, j] = text.split().count(word)
    else:
        raise ValueError("Uknown vectorization method")
    
    return matrix


def main():
    # Load news and aggragate
    news_df = pd.read_csv(DATA_DIR / "all_news.csv")
    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
    news_agg = aggregate_news(news_df)
    news_agg.to_csv(DATA_DIR / "aggregated_news.csv", index=False)
    print("Created aggregated_news.csv")

    # laod impact scores and fix date types
    impact_df = pd.read_csv(DATA_DIR / "historical_prices_impact.csv")
    impact_df["date"] = pd.to_datetime(impact_df["date"], errors="coerce")

    # Normalize timezones, doing this because we were getting a timezeon issue
    news_agg["date"] = news_agg["date"].dt.tz_localize(None)
    impact_df["date"] = impact_df["date"].dt.tz_localize(None)

    # Merge by date and symbol
    merged_df = pd.merge(news_agg, impact_df, on=["date", "symbol"], how="left")
    print("Merged with impact scores")
    
    # Vectorize using all three methods
    for method in ["dtm", "tfidf", "curated"]:
        matrix = vectorize_news(merged_df, method=method)
        out_df = merged_df.copy()

        if method != "curated":
            out_df["news_vector"] = list(matrix.toarray())
        else:
            out_df["news_vector"] = list(matrix.tolist())

        out_df = out_df[["date", "symbol", "news_vector", "impact_score"]]
        out_path = DATA_DIR / f"vectorized_news_{method}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"Created {out_path.name}")

    print("\nAll vectorized datasets created successfully!")

if __name__ == "__main__":
    main()

