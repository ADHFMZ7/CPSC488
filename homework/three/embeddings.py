from gensim.utils import simple_preprocess
import pandas as pd
from enum import Enum
from torch import nn
import torch
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np

from pathlib import Path

DATA_PATH = Path('datasets')

class Method(Enum):
    CBOW = 0
    SG = 1


def generate_embeddings(df: pd.DataFrame, method: Method, embed_dim: int = 10, min_count=4):
    '''Generate vectorized articles and append impact scores to create datset'''

    prices = pd.read_csv(DATA_PATH/'historical_prices_impact.csv', parse_dates=['date'])
    merged_df = pd.merge(df ,prices, how='inner', on=['date', 'symbol'])

    merged_df = merged_df.dropna()

    merged_df['tokens'] = merged_df.news.apply(lambda x: [t for t in simple_preprocess(x) if t not in STOPWORDS])

    sentences = [tokens for tokens in merged_df.tokens if len(tokens)]

    if method == Method.CBOW:
        raise NotImplementedError('Method not implemented')

    elif method == Method.SG:
        model = Word2Vec(sentences,sg=1, vector_size=embed_dim, min_count=min_count, workers=4)

        vectorize = lambda tokens: np.stack([model.wv[token] for token in tokens if token in model.wv]).mean(axis=0)
        merged_df['news_vector'] = merged_df.tokens.apply(vectorize)

    else:
        raise ValueError('Unknown embedding method')


    return merged_df [['date', 'symbol', 'news_vector', 'impact_score']]


def train_cbow():
    pass



class CBOW(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass        


