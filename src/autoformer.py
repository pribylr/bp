import tensorflow as tf
import numpy as np

class Autocorrelation(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Autocorrelation, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']

        self.query = tf.keras.layers.Dense(
            config['d_k'], 
            kernel_initializer='glorot_uniform', 
            bias_initializer='glorot_uniform')
        self.key = tf.keras.layers.Dense(
            config['d_k'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
        self.value = tf.keras.layers.Dense(
            config['d_v'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')

    def call(self, input):
        queries = self.query(input[0])
        keys = self.query(input[1])
        values = self.query(input[2])
        
        

class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.autocorrelation = Autocorrelation(config)

    def call(self, input):
        return input
        
class Decoder(tf.keras.layers.Layer):
    def __init__(self, config,**kwargs):
        super(Decoder, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.autocorrelation1 = Autocorrelation(config)
        self.autocorrelation2 = Autocorrelation(config)
    
    def call(self, input):
        return input

class Autoformer(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(Autoformer, self).__init__()
        self.input_seq_len = config['input_seq_len']
        self.O = config['O']
        self.pool_size = config['pool_size']
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

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
