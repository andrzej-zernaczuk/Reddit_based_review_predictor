import numpy as np
import pandas as pd
from RedditDataPreparation.RedditApiConnector import RedditApiConnector
from RedditDataPreparation.DataPreprocessing import DataPreprocesser
from Models.FirstCNN import FirstCNN
import logging

logging.getLogger('tensorflow').setLevel(logging.WARNING)


class RedditBasedPredictor:

    def __init__(self, name, subreddit='movies'):
        self.subreddit = subreddit
        self.model = FirstCNN()
        self.connector = RedditApiConnector(self.subreddit)
        self.data = self.get_data(name)
        self.preprocessor = DataPreprocesser(self.data, 'body')

    def get_data(self, name):
        return self.connector.search_comments(name)

    def make_prediction(self):
        clean_data = self.preprocessor.full_prepare_data(self.data, 'body')
        return self.model.predict_sentiment(clean_data)

    def prepare_avg(self):
        preds = self.make_prediction()
        weights_reshape = np.reshape(self.data['score'].values, (preds.shape))
        return np.average(preds, weights=weights_reshape) * 100 

class TestPredictor:
    def __init__(self):
        self.data = pd.read_csv('Data/titles_with_reviews.csv')
        self.data['preds'] = np.NaN
        self.data['preds_diff'] = np.NaN

    def make_preds(self):
        for i, movie in enumerate(self.data['original_title']):
            try:
                predictor = RedditBasedPredictor(movie)
                prediction = predictor.prepare_avg()
                self.data['preds'].iloc[i] = prediction * 100
                self.data['preds_diff'] = np.absolute(self.data['preds']  - self.data['preds'])
                print(self.data.iloc[i])
                if i % 5 == 0:
                    self.data.to_csv('preds.csv', mode='a')
            except KeyError:
                print('No reddit comments')
                
class DataHandling:
    
    def __init__(self):
        self.data = self.preparing_data()
        
    def preparing_data(self):
        top50 = pd.read_csv('Data/Filmweb_top50.csv', index_col=0)
        top50['imdb'] = (top50['imdb'].str.replace('/10', '').astype(float)) * 10 
        top50['rottenTomatoes'] = top50['rottenTomatoes'].str.replace('%', '').astype(float) 
        top50['metacritic'] = top50['metacritic'].str.replace('/100', '').astype(float) 

        return top50

    def measure_one_movie(self, titleFromPrediction):
        predictor = RedditBasedPredictor(titleFromPrediction)
        resultFromPrediction = predictor.prepare_avg()
        
        errorIMDB = round(abs(resultFromPrediction - self.data.loc[titleFromPrediction, 'imdb']), 2)
        errorRotten = round(abs(resultFromPrediction - self.data.loc[titleFromPrediction, 'rottenTomatoes']), 2)
        errorMetacritic = round(abs(resultFromPrediction - self.data.loc[titleFromPrediction, 'metacritic']), 2)

        data = [titleFromPrediction, self.data.loc[titleFromPrediction, 'year'], resultFromPrediction,
                self.data.loc[titleFromPrediction, 'imdb'], errorIMDB, self.data.loc[titleFromPrediction, 'rottenTomatoes'],
                errorRotten, self.data.loc[titleFromPrediction, 'metacritic'], errorMetacritic]
        print(data)
        return data
    

    def measure_all_movies(self):
    
        full_list = []
        columns = ['title', 'year', 'prediction', 'imdb', 'errorIMDB', 'rottenTomatoes', 'errorRotten', 'metacritic',
                   'errorMetacritic']
        data_df = pd.DataFrame(columns=columns)
        for i, movie in enumerate(self.data.index):
            print(i, movie)
            data_df.loc[len(data_df.index)] = self.measure_one_movie(movie)
            if i % 5 == 0:
                data_df.to_csv('top_50_preds.csv', mode='a')
        data_df.to_csv('top_50_preds.csv', mode='a')

    
if __name__ == '__main__':
    FinalTest = DataHandling()
    FinalTest.measure_all_movies()
