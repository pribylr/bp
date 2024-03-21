import tensorflow as tf
import numpy as np

class Autoformer(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(TFT, self).__init__()
        self.avg_pool1d = tf.keras.layers.AvgPool1D(pool_size=pool_size, strides=1, padding='same', data_format='channels_last')

    def series_decomp(self, X, pool_size: int):
        X_t = self.avg_pool1d(X)
        X_t = tf.cast(X_t, tf.float64)
        X_s = X - X_t
        return X_t, X_s
        
    def call(self, input):
        # ...
        return input
