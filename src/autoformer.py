import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

    def call(self, input):
        return input
        
class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

    def call(self, input):
        return input

class Autoformer(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(Autoformer, self).__init__()
        self.input_seq_len = config['input_seq_len']
        self.O = config['O']
        self.pool_size = config['pool_size']
        self.encoder = Encoder()
        self.decoder = Decoder()

    
    def series_decomp(self, X):
        avg_pool1d = tf.keras.layers.AvgPool1D(pool_size=self.pool_size, strides=1, padding='same', data_format='channels_last')
        X_t = avg_pool1d(X)
        X_s = X - X_t
        return X_t, X_s
        
    def prepare_input(self, input):
        X_ent, X_ens = self.series_decomp(input[:, (input.shape[1])//2:, :])
        mean = tf.reduce_mean(input, axis=1, keepdims=True)
        mean = tf.tile(mean, [1, self.O, 1])
        zeros = tf.zeros([input.shape[0], self.O, input.shape[2]], dtype=tf.float32)

        X_det = tf.concat([X_ent, mean], axis=1)
        X_des = tf.concat([X_ens, zeros], axis=1)
        return X_det, X_des
    
    def call(self, input):
        X_det, X_des = self.prepare_input(input)        
        enc_out = self.encoder(input)
        dec_out = self.decoder((enc_out, X_det, X_des))
        return dec_out
