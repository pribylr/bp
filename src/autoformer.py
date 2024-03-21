import tensorflow as tf
import numpy as np

class Series_decomp(tf.keras.layers.Layer):  # ?
    def __init__(self, config, **kwargs):
        super(Series_decomp, self).__init__()
        self.pool_size = config['pool_size']
        self.avg_pool1d = tf.keras.layers.AvgPool1D(pool_size=config['pool_size'], strides=1, padding='same', data_format='channels_last')
        
    def call(self, input):
        X_t = self.avg_pool1d(input)
        X_s = input - X_t
        return X_t, X_s

class Autocorrelation(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Autocorrelation, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.heads = config['ac_heads']
        
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

    def createQKV(self, input):
        queries = [self.query(input[0] for _ in range(self.heads)])  # [(batch, seq_len, features), ...]
        keys = [self.query(input[1] for _ in range(self.heads)])
        values = [self.query(input[2] for _ in range(self.heads)])
        Q_4d = tf.stack(queries, axis=-1)  # (batch, seq_len, features, heads)
        K_4d = tf.stack(keys, axis=-1)
        V_4d = tf.stack(values, axis=-1)
        Q_4d = tf.transpose(Q_4d, perm=[0, 3, 2, 1])  # (batch, heads, features, seq_len)
        K_4d = tf.transpose(K_4d, perm=[0, 3, 2, 1])
        V_4d = tf.transpose(V_4d, perm=[0, 3, 2, 1])
        return Q_4d, K_4d, V_4d
    
    def time_delay_agg(self, input):
        
        return input
        
    def call(self, input):
        Q, K, V = self.createQKV(input)  # (batch, heads, features, seq_len)
        Q = tf.signals.fft(Q)  # (batch, heads, features, seq_len)
        K = tf.signals.fft(K)  # (batch, heads, features, seq_len)
        K = tf.math.conj(K)
        QK = tf.multiply(Q, K)
        QK = tf.signal.ifft(QK)
        delay = self.time_delay_agg((QK, V))
        return delay
        

class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.autocorrelation = Autocorrelation(config)

    def call(self, input):
        ac = self.autocorrelation((input, input, input))
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
        self.series_decomp = Series_decomp(config)
        
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
