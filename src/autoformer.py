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
        self.c = config['c']
        self.input_seq_len = config['input_seq_len']
        
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
        queries = [self.query(input[0]) for _ in range(self.heads)]  # [(batch, seq_len, d_k), ...]
        keys = [self.query(input[1]) for _ in range(self.heads)]  # [(batch, seq_len, d_k), ...]
        values = [self.query(input[2]) for _ in range(self.heads)]  # [(batch, seq_len, d_v), ...]
        Q_4d = tf.stack(queries, axis=-1)  # (batch, seq_len, d_k, heads)
        K_4d = tf.stack(keys, axis=-1)  # (batch, seq_len, d_k, heads)
        V_4d = tf.stack(values, axis=-1)  # (batch, seq_len, d_v, heads)
        Q_4d = tf.cast(Q_4d, tf.complex64)
        K_4d = tf.cast(K_4d, tf.complex64)
        #V_4d = tf.cast(V_4d, tf.complex64)
        Q_4d = tf.transpose(Q_4d, perm=[0, 3, 2, 1])  # (batch, heads, d_k, seq_len)
        K_4d = tf.transpose(K_4d, perm=[0, 3, 2, 1])  # (batch, heads, d_k, seq_len)
        V_4d = tf.transpose(V_4d, perm=[0, 3, 2, 1])  # (batch, heads, d_v, seq_len)
        return Q_4d, K_4d, V_4d
    
    def time_delay_agg(self, input):  # (batch, heads, d_k, seq_len) (batch, heads, d_v, seq_len)
        qk_abs = tf.abs(input[0])
        k = int(self.c*np.log(input[0].shape[-1]))
        values, indices = tf.math.top_k(qk_abs, k=k)  # (batch, heads, d_k, k)
        values_sm = tf.nn.softmax(values)
        def roll_each_feature(value_matrix, lag):  # (heads, d_k, seq_len) (heads, d_k)
            rolled = tf.TensorArray(dtype=value_matrix.dtype, size=self.heads*self.d_v)
            idx = 0
            for head in range(self.heads):
                for feature in range(self.d_v):
                    current_lag = lag[head, feature]  # index nejvyssi hodnoty pro konkretni batch, head, feature
                    v_slice = value_matrix[head, feature, :]  # (seq_len, )
                    rolled_v_slice = tf.roll(v_slice, shift=current_lag, axis=0)  # (seq_len, )
                    rolled = rolled.write(idx, rolled_v_slice)
                    idx += 1
            result = tf.reshape(rolled.stack(), (self.heads, self.d_v, self.input_seq_len))  # (heads, d_v, seq_len)
            return result
            
        time_delay_aggregated = []
        for i in range(k):
            # highest values indices for all batches, heads, features, for day k
            current_lag = -indices[..., i]  # (batch, heads, d_k)
            print('  current_lag:', current_lag.shape)
            rolled_v =tf.map_fn(
                fn=lambda elems: roll_each_feature(elems[0], elems[1]),
                elems=(input[1], current_lag),
                dtype=input[1].dtype
            )  # (batch, heads, d_v, seq_len)
            weighted_rolled_V = rolled_v * tf.expand_dims(values_sm[..., i], axis=-1)  # (batch, heads, d_v, seq_len)
            time_delay_aggregated.append(weighted_rolled_V)
            
        stacked = tf.stack(time_delay_aggregated, axis=-1)  # (batch, heads, d_v, seq_len, k)
        aggregated_representation = tf.reduce_sum(tf.stack(time_delay_aggregated, axis=-1), axis=-1)  # (batch, heads, d_v, seq_len)
        reshaped = tf.reshape(aggregated_representation, (aggregated_representation.shape[0], aggregated_representation.shape[1]*aggregated_representation.shape[2], aggregated_representation.shape[3]))  # (batch, heads*d_v, seq_len)
        res = tf.transpose(reshaped, perm=[0, 2, 1])  # (batch, seq_len, d_v*heads)
        return res
        
    def call(self, input):
        Q, K, V = self.createQKV(input)  # (batch, heads, d_k, seq_len)
        Q = tf.signal.fft(Q)  # (batch, heads, features, seq_len)
        K = tf.signal.fft(K)  # (batch, heads, features, seq_len)
        K = tf.math.conj(K)
        QK = tf.multiply(Q, K)  # (batch, heads, d_k, seq_len)
        QK = tf.signal.ifft(QK)
        delay = self.time_delay_agg((QK, V))  # (batch, seq_len, d_v*heads)
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
