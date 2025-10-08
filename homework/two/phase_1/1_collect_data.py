import pandas as pd

headlines = pd.read_csv('../../../data/news_datasets/headlines.csv')
ratings = pd.read_csv('../../../data/news_datasets/analyst_ratings.csv')

print(headlines.head())
print(ratings.head())


