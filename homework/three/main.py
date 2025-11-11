import pandas as pd
from pathlib import Path

from embeddings import generate_embeddings, Method
from train import train_model

PATH = Path('datasets/')

def main():

    agg = pd.read_csv(PATH/'aggregated_news.csv', parse_dates=['date'])

    # Generate embeddings
    sg_embeddings = generate_embeddings(agg, Method.SG)
    sg_embeddings.to_csv(PATH/'vectorized_news_skipgram_embeddings.csv')

    # sg_embeddings = pd.read_csv(PATH/'vectorized_news_skipgram_embeddings.csv', parse_dates=['date'])

    # cbow_embeddings = generate_embeddings(agg, Method.CBOW)

    # Train both models
    train_model(sg_embeddings, 'v0-skipgram')

    # Evaluate the models

    # Print metrics for both models

if __name__ == "__main__":
    main()
