import multiprocessing
import pandas as pd
import gensim.downloader as api
from gensim.models import Word2Vec

print(multiprocessing.cpu_count())

tokens = pd.Series()
tweets = pd.read_csv('digikala_comments.csv', low_memory=False)

# drop nan records
for index in range(tweets.shape[0]):
    tokens = tokens.append(pd.Series([tweets.loc[index].dropna().tolist()]), ignore_index=True)

tokens_list = tokens.tolist()
w2vec = Word2Vec(sentences=tokens_list, size=100, window=3, min_count=10, negative=3, iter=100,
                 workers=multiprocessing.cpu_count())
w2vec.wv.save_word2vec_format('digikala_comments.bin', binary=False)

