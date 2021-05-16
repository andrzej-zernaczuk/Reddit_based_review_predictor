import tensorflow as tf
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pickle

class DataPreprocesser:

    def __init__(self, data, text_col):

        self.data = data
        self.text_col = text_col
        # nltk.download('stopwords')

    @staticmethod
    def initial_data_prep(text):
        text = text.lower()
        # Replace new rows with space
        text = text.replace('\n', " ").replace("\r", " ")
        # Create list of all not needed chars
        punc_list = '!"@#$%^&*()+_-.<>?/:;[]{}|\~'
        # Make transformation with dict that contains punc_list chars
        t = str.maketrans(dict.fromkeys(punc_list, " "))
        # Apply transformation
        text = text.translate(t)
        # Replace single quote with empty char
        t = str.maketrans(dict.fromkeys("'`", ""))
        text = text.translate(t)

        return text

    @staticmethod
    def remove_stop_words(text):
        # Prepare set of stopwords
        stop_words = set(stopwords.words('english'))
        # Remove stopwords from the text
        filtered_text = [word for word in text.split() if not word in stop_words]

        return filtered_text

    @staticmethod
    def text_stemming(text):
        stemmer = nltk.porter.PorterStemmer()
        text = [stemmer.stem(word) for word in text]
        return text

    def clean_data(self, data, text_col):
        data = data.copy()
        data[text_col] = data[text_col].apply(lambda x: self.initial_data_prep(x)).copy()
        # Apply func that removes stop words
        data[text_col] = data[text_col].apply(lambda x: self.remove_stop_words(x)).copy()
        # Apply func that performs lemmatization
        data[text_col] = data[text_col].apply(lambda x: self.text_stemming(x))

        return data


    def tokenize(self, data, text_col, num_words_pad=100):
        # Initialize tokenizer
        with open('Data/tokenizer.pickle', 'rb') as handle:
            tok = pickle.load(handle)
        seq = tok.texts_to_sequences(list(data[text_col]))
        # Pad sequences to make them same lenght
        tf_ready = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=num_words_pad)
        # Prepare dataframe ready for training
        tf_df = pd.DataFrame(tf_ready)

        return tf_df

    def full_prepare_data(self, data, text_col):
        # Clean the data
        clean_data = self.clean_data(data, text_col)
        # Return tokenized data
        return self.tokenize(clean_data, text_col)