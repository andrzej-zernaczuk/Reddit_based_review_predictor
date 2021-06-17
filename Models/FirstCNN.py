import tensorflow as tf
class FirstCNN:

    def __init__(self):
        self.model = tf.keras.models.load_model('rnn')
        """
        Model architecture:
        model = Sequential(
            [
            tf.keras.layers.Conv1D(filters=32, 
                kernel_size=10,
                strides=10,
                padding='valid',
                activation='relu'),
                tf.keras.layers.MaxPool1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='sigmoid'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ]
        )
        Layer (type)                 Output Shape              Param #   
        =================================================================
        embedding_3 (Embedding)      (None, 100, 32)           320000    
        _________________________________________________________________
        conv1d_4 (Conv1D)            (None, 10, 32)            10272     
        _________________________________________________________________
        max_pooling1d_3 (MaxPooling1 (None, 5, 32)             0         
        _________________________________________________________________
        flatten_3 (Flatten)          (None, 160)               0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 10)                1610      
        _________________________________________________________________
        dense_2 (Dense)              (None, 1)                 11        
        =================================================================
        Total params: 331,893
        Trainable params: 331,893
        Non-trainable params: 0
        """

    def predict_sentiment(self, tf_df):
        print(tf_df)
        print(tf_df.shape)
        predictions = self.model.predict(tf_df)
        return predictions

