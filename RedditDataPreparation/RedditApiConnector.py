import pandas as pd
from psaw import PushshiftAPI
import numpy as np

class RedditApiConnector:

    def __init__(self, name, subreddit):

        self.name = name
        self.subreddit = subreddit
        self.api = PushshiftAPI()

    def search_comments(self, limit=1000):

        # Hardcode filters to keep same columns in dataframes
        filter = ['author', 'date', 'title', 'body', 'score']

        # Connect to api, and get data
        comments = self.api.search_comments(
            subreddit = self.subreddit,
            filter=filter, q=self.name, limit=limit
        )
        # Prepare dataframe
        df = pd.DataFrame([comment.d_ for comment in comments])

        return df

    def search_submissions(self, limit=1000):
        # Hardcode filters to keep same columns in dataframes
        filer = ['author', 'date', 'title', 'selftext', 'score']

        submissions = self.api.search_submissions(
            subreddit=self.subreddit,
            filer=filer, q=self.name, limit=limit
        )

        df = pd.DataFrame([submission.d_ for submission in submissions])
        # Rename columns to keep all dataframes same
        df.rename(columns={'selftext': 'body'}, inplace=True)
        # Replace '[removed]' with NaN values
        df['body'].replace(['removed'], np.NaN, inplace=True)

        return df

a = RedditApiConnector(name='Star Wars', subreddit='movies')
print(a.search_comments())