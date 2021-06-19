import numpy as np
import pandas as pd
from RedditDataPreparation.RedditApiConnector import RedditApiConnector
from RedditDataPreparation.DataPreprocessing import DataPreprocesser
from Models.FirstCNN import FirstCNN
import time


class RedditBasedPredictor:

    def __init__(self, name, subreddit='movies'):
        self.name = name
        self.subreddit = subreddit
        self.model = FirstCNN()
        self.connector = RedditApiConnector(self.name, self.subreddit)
        self.data = self.get_data()
        self.preprocessor = DataPreprocesser(self.data, 'body')

    def get_data(self):
        return self.connector.search_comments()

    def make_prediction(self):
        clean_data = self.preprocessor.full_prepare_data(self.get_data(), 'body')
        return self.model.predict_sentiment(clean_data)

    # Pomyslec nad wagami takimi zeby wywalalo te z ujemnym scorem
    def prepare_avg(self, preds):
        weights_reshape = np.reshape(self.data['score'].values, (preds.shape))
        return np.average(preds, weights=weights_reshape)


class DataHandling:

    def preparing_data(self):
        top50 = pd.read_csv('Filmweb_top50.csv', index_col=0)
        top50['imdb'] = (top50['imdb'].str.replace('/10', '').astype(float)) / 10
        top50['rottenTomatoes'] = top50['rottenTomatoes'].str.replace('%', '').astype(float) / 100
        top50['metacritic'] = top50['metacritic'].str.replace('/100', '').astype(float) / 100

        columns = ['title', 'year', 'prediction', 'imdb', 'errorIMDB', 'rottenTomatoes', 'errorRotten', 'metacritic',
                     'errorMetacritic']
        result_df = pd.DataFrame(columns=columns)
        result_df = result_df.set_index('title')

    def error_measuring(self):
        titleFromPrediction = "Joker"
        resultFromPrediction = 0.90

        errorIMDB = round(abs((resultFromPrediction - top50.loc[titleFromPrediction, 'imdb']) * 100), 2)
        errorRotten = round(abs((resultFromPrediction - top50.loc[titleFromPrediction, 'rottenTomatoes']) * 100), 2)
        errorMetacritic = round(abs((resultFromPrediction - top50.loc[titleFromPrediction, 'metacritic']) * 100), 2)

        data = [titleFromPrediction, top50.loc[titleFromPrediction, 'year'], resultFromPrediction,
                top50.loc[titleFromPrediction, 'imdb'], errorIMDB, top50.loc[titleFromPrediction, 'rottenTomatoes'],
                errorRotten, top50.loc[titleFromPrediction, 'metacritic'], errorMetacritic]

        columns = ['title', 'year', 'prediction', 'imdb', 'errorIMDB', 'rottenTomatoes', 'errorRotten', 'metacritic',
                   'errorMetacritic']

        data_df = pd.DataFrame([data], columns=columns)
        data_df = data_df.set_index('title')
        result_df = result_df.append(data_df)

# XDDDD TE WYNIKI TAKIE NIE ZA DOBRE
# ZMIENIC DLUGOSC TENSORA PO TOKENIZACJI BO W TRENOWANYM DATASECIE BYLY PONAD 100 A TUTAJ SREDNIA TO 48
# OGRNAC Z POSTOW A NIE KOMENTARZY TYLKO
# Z 40 COS NIE DZIALA, MOZE PRZY ZA DLUGICH TRZEBA ZOBACZYC
if __name__ == '__main__':
    start = time.time()
    reddit = RedditBasedPredictor('Joker')
    preds = reddit.make_prediction()
    print(reddit.prepare_avg(preds))
    print(f'---------TIME: {time.time() - start}---------')
