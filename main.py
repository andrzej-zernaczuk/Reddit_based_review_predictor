import numpy as np
import pandas as pd
from RedditDataPreparation.RedditApiConnector import RedditApiConnector
from RedditDataPreparation.DataPreprocessing import DataPreprocesser
from Models.FirstCNN import FirstCNN
import time


class RedditBasedPredictor:

    def __init__(self, subreddit='movies'):
        self.subreddit = subreddit
        self.model = FirstCNN()
        self.connector = RedditApiConnector(self.subreddit)
        self.data = self.get_data()
        self.preprocessor = DataPreprocesser(self.data, 'body')

    def get_data(self, name):
        return self.connector.search_comments(name)

    def make_prediction(self, name):
        clean_data = self.preprocessor.full_prepare_data(self.get_data(name), 'body')
        return self.model.predict_sentiment(clean_data)

    # Pomyslec nad wagami takimi zeby wywalalo te z ujemnym scorem
    def prepare_avg(self, name):
        preds = self.make_prediction(name)
        weights_reshape = np.reshape(self.data['score'].values, (preds.shape))
        return np.average(preds, weights=weights_reshape)


class ErrorMeasuring:

    def preparing_data(self):
        global df
        df = pd.read_json('Filmweb_top50.json')



if __name__ == '__main__':
    start = time.time()
    reddit = RedditBasedPredictor()
    print(reddit.prepare_avg('Joker'))
    print(f'---------TIME: {time.time() - start}---------')
