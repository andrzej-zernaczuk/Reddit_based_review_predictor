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
        global df
        df = pd.read_csv('Filmweb_top50.csv', index_col=0)
        df['imdbRating'] = (df['imdbRating'].str.replace('/10', '').astype(float)) / 10
        df['rottenTomatoes'] = df['rottenTomatoes'].str.replace('%', '').astype(float) / 100
        df['metacritic'] = df['metacritic'].str.replace('/100', '').astype(float) / 100

    def error_measuring(self):
        titleFromPrediction = "Joker"
        scoreFromPrediction = 0.90

        errorIMDB = round(abs((scoreFromPrediction - df.loc[titleFromPrediction, 'imdbRating']) * 100), 2)
        errorRotten = round(abs((scoreFromPrediction - df.loc[titleFromPrediction, 'rottenTomatoes']) * 100), 2)
        errorMetacritic = round(abs((scoreFromPrediction - df.loc[titleFromPrediction, 'metacritic']) * 100), 2)

        print(errorIMDB)
        print(errorRotten)
        print(errorMetacritic)


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