import pandas as pd
from psaw import PushshiftAPI

class RedditApiConnector:

    def __init__(self, name, subreddit):

        self.name = name
        self.subreddit = subreddit
        self.api = PushshiftAPI()

    def search_comments(self, limit=1000, **kwargs):

        # Hardcode filters to keep same columns in dataframes
        filter = ['author', 'date', 'title', 'body', 'score']

        # Connect to api, and get data
        comments = self.api.search_comments(kwargs,
            subreddit = self.subreddit, filter=filter,
            q=self.name, limit=limit
        )
        # Prepare dataframe
        df = pd.DataFrame([comment.d_ for comment in comments])

        return df
