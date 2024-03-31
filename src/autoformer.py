import tensorflow as tf
import numpy as np

class Series_decomp(tf.keras.layers.Layer):  # ?
    def __init__(self, config, **kwargs):
        super(Series_decomp, self).__init__()
        self.pool_size = config['pool_size']
        self.avg_pool1d = tf.keras.layers.AvgPool1D(pool_size=config['pool_size'], strides=1, padding='same', data_format='channels_last')
        
    def call(self, input):  # (batch, seq_len, features)
        X_t = self.avg_pool1d(input)  # (batch, seq_len, features)
        X_s = input - X_t  # (batch, seq_len, features)
        return X_s, X_t


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, config, d_out, **kwargs):
        super(FeedForward, self).__init__()
        
        #self.d_model = config['d_model']
        #self.d_ff = config['d_ff']
        #self.dropout_rate = config['dropout_rate']
        #self.training = config['training']
        
        self.fc1 = tf.keras.layers.Dense(config['d_ff'], activation=config['activation'])
        self.dropout1 = tf.keras.layers.Dropout(config['dropout_rate'])
        self.fc2 = tf.keras.layers.Dense(d_out, activation=config['activation'])
        self.dropout2 = tf.keras.layers.Dropout(config['dropout_rate'])

    def call(self, input, training):
        x = self.fc1(input)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return x
        

class Autocorrelation(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Autocorrelation, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.d_model = config['d_model']
        self.heads = config['ac_heads']
        self.c = config['c']
        self.input_seq_len = config['input_seq_len']
        
        self.query = tf.keras.layers.Dense(
            config['d_model']/config['ac_heads'], 
            kernel_initializer='glorot_uniform', 
            bias_initializer='glorot_uniform')
        self.key = tf.keras.layers.Dense(
            config['d_model']/config['ac_heads'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
        self.value = tf.keras.layers.Dense(
            config['d_model']/config['ac_heads'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
        self.out = tf.keras.layers.Dense(
            config['d_model'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')

    def createQKV(self, input):
        Q = input[0]
        K = input[1]
        V = input[2]
        if input[0].shape[1] > input[1].shape[1]:  # Q longer seq_len -- zero filling
            paddings = tf.constant([[0, 0], [0, Q.shape[1] - K.shape[1]], [0, 0]])
            K = tf.pad(K, paddings)
            V = tf.pad(V, paddings)
        if input[0].shape[1] < input[1].shape[1]:  # Q shorter seq_len -- truncation
            len = Q.shape[1]
            K = K[:, -len:, :]
            V = V[:, -len:, :]
            
        queries = [self.query(Q) for _ in range(self.heads)]  # [(batch, seq_len, d_model/heads), ...]
        keys = [self.query(K) for _ in range(self.heads)]  # [(batch, seq_len, d_model/heads), ...]
        values = [self.query(V) for _ in range(self.heads)]  # [(batch, seq_len, d_model/heads), ...]
        Q_4d = tf.stack(queries, axis=-1)  # (batch, seq_len, d_model/heads, heads)
        K_4d = tf.stack(keys, axis=-1)  # (batch, seq_len, d_model/heads, heads)
        V_4d = tf.stack(values, axis=-1)  # (batch, seq_len, d_model/heads, heads)
        
        Q_4d = tf.cast(Q_4d, tf.complex64)
        K_4d = tf.cast(K_4d, tf.complex64)
        Q_4d = tf.transpose(Q_4d, perm=[0, 3, 2, 1])  # (batch, heads, d_model/heads, seq_len)
        K_4d = tf.transpose(K_4d, perm=[0, 3, 2, 1])  # (batch, heads, d_model/heads, seq_len)
        V_4d = tf.transpose(V_4d, perm=[0, 3, 2, 1])  # (batch, heads, d_model/heads, seq_len)
        return Q_4d, K_4d, V_4d
    
    def time_delay_agg(self, input):  # (batch, heads, d_model/heads, seq_len) (batch, heads, d_model/heads, seq_len)
        k = int(self.c*np.log(input[0].shape[-1]))
        values, indices = tf.math.top_k(input[0], k=k)
        values_sm = tf.nn.softmax(values)
        def roll_each_feature(value_matrix, lag):  # (heads, d_model/heads, seq_len) (heads, d_model/heads)
            rolled = tf.TensorArray(dtype=value_matrix.dtype, size=self.d_model)
            idx = 0
            for head in range(self.heads):
                for feature in range(self.d_v):
                    current_lag = lag[head, feature]  # index nejvyssi hodnoty pro konkretni batch, head, feature
                    v_slice = value_matrix[head, feature, :]  # (seq_len, )
                    rolled_v_slice = tf.roll(v_slice, shift=current_lag, axis=0)  # (seq_len, )
                    rolled = rolled.write(idx, rolled_v_slice)
                    idx += 1
            result = tf.reshape(rolled.stack(), (self.heads, int(self.d_model/self.heads), input[0].shape[-1]))  # (heads, d_model/heads, seq_len)
            return result
            
        time_delay_aggregated = []
        for i in range(k):
            # highest values indices for all batches, heads, features, for day k
            current_lag = -indices[..., i]  # (batch, heads, d_model/heads)
            rolled_v =tf.map_fn(
                fn=lambda elems: roll_each_feature(elems[0], elems[1]),
                elems=(input[1], current_lag),
                dtype=input[1].dtype
            )  # (batch, heads, d_model/heads, seq_len)
            weighted_rolled_V = rolled_v * tf.expand_dims(values_sm[..., i], axis=-1)  # (batch, heads, d_model/heads, seq_len)
            time_delay_aggregated.append(weighted_rolled_V)
            
        stacked = tf.stack(time_delay_aggregated, axis=-1)  # (batch, heads, d_model/heads, seq_len, k)
        aggregated_representation = tf.reduce_sum(tf.stack(time_delay_aggregated, axis=-1), axis=-1)  # (batch, heads, d_model/heads, seq_len)
        
        reshaped = tf.reshape(aggregated_representation, (tf.shape(aggregated_representation)[0], aggregated_representation.shape[1]*aggregated_representation.shape[2], aggregated_representation.shape[3]))  # (batch, heads*d_model/heads, seq_len)
        res = tf.transpose(reshaped, perm=[0, 2, 1])  # (batch, seq_len, d_model)
        return res
        
    def call(self, input):
        Q, K, V = self.createQKV(input)  # (batch, heads, d_model/heads, seq_len)
        Q = tf.signal.fft(Q)  # (batch, heads, d_model/heads, seq_len)
        K = tf.signal.fft(K)  # (batch, heads, d_model/heads, seq_len)
        K = tf.math.conj(K)
        QK = tf.multiply(Q, K)  # (batch, heads, d_model/heads, seq_len)
        QK = tf.signal.ifft(QK)
        
        delay = self.time_delay_agg((tf.math.real(QK), V))  # (batch, seq_len, d_model)
        return delay
        

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(EncoderLayer, self).__init__()
        self.autocorrelation = Autocorrelation(config)
        self.series_decomp1 = Series_decomp(config)
        self.series_decomp2 = Series_decomp(config)
        self.feed_forward = FeedForward(config, config['d_model'])
        self.linear1 = tf.keras.layers.Dense(
            config['d_model'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
    
    def call(self, input, training):  # (batch, seq_len, d_model)
        x = self.autocorrelation((input, input, input))  # (batch, seq_len, d_model)
        x += input
        S1, _ = self.series_decomp1(x)
        x = self.feed_forward(S1, training)
        x += S1
        S2, _ = self.series_decomp2(x) 
        return S2  # (batch, seq_len, d_model)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.encoder_layers_num = config['encoder_layers']
        self.encoder_layers = [EncoderLayer(config) for _ in range(config['encoder_layers'])]
        self.embed = tf.keras.layers.Dense(
            config['d_model'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')

    def call(self, input, training):
        x = self.embed(input)  # (batch, seq_len, d_model)
        for i in range(self.encoder_layers_num):
            x = self.encoder_layers[i](x, training)  # (batch, seq_len, d_model)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwarfs):
        super(DecoderLayer, self).__init__()
        self.autocorrelation1 = Autocorrelation(config)
        self.autocorrelation2 = Autocorrelation(config)
        self.series_decomp1 = Series_decomp(config)
        self.series_decomp2 = Series_decomp(config)
        self.series_decomp3 = Series_decomp(config)
        self.feed_forward = FeedForward(config, config['d_model'])
        self.mlp1 = FeedForward(config, config['d_out'])
        self.mlp2 = FeedForward(config, config['d_out'])
        self.mlp3 = FeedForward(config, config['d_out'])

    def call(self, input, training):
        # [0] -- embed S
        # [1] -- T
        # [2] -- enc out
        x = self.autocorrelation1((input[0], input[0], input[0]))  # (batch, seq_len/2 + O, d_model)
        x += input[0]
        S1, T1 = self.series_decomp1(x)
        y = self.autocorrelation2((S1, input[2], input[2]))  # (batch, seq_len/2 + O, d_model)
        y += S1
        S2, T2 = self.series_decomp2(y)
        z = self.feed_forward(S2, training)
        z += S2
        S3, T3 = self.series_decomp3(z)
        
        T = input[1]
        T += self.mlp1(T1, training)
        T += self.mlp2(T2, training)
        T += self.mlp3(T3, training)
        return S3, T
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__()
        self.decoder_layers_num = config['decoder_layers']
        self.decoder_layers = [DecoderLayer(config) for _ in range(config['decoder_layers'])]
        self.embed = tf.keras.layers.Dense(
            config['d_model'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
        self.mlp = FeedForward(config, config['d_out'])
    
    def call(self, input, training):  # X_des (batch, seq_len/2 + O, features), X_det (batch, seq_len/2 + O, features), enc_out (batch, seq_len, d_model)
        S = self.embed(input[0])  # (batch, seq_len/2 + O, d_model)
        T = input[1]
        for i in range(self.decoder_layers_num):
            S, T = self.decoder_layers[i]((S, T, input[2]), training)
        
        S = self.mlp(S, training)
        return S+T

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
        
    def prepare_input(self, input):  # (batch, seq_len, features)
        X_ens, X_ent = self.series_decomp(input[:, (input.shape[1])//2:, :])  # (batch, seq_len/2, features)
        mean = tf.reduce_mean(input, axis=1, keepdims=True)
        mean = tf.tile(mean, [1, self.O, 1])  # (batch, output_seq_len, features)
        zeros = tf.zeros([tf.shape(input)[0], self.O, input.shape[2]], dtype=tf.float32)  # (batch, output_seq_len, features)
        
        X_det = tf.concat([X_ent, mean], axis=1)  # (batch, seq_len/2 +output_seq_len, features)
        X_des = tf.concat([X_ens, zeros], axis=1)  # (batch, seq_len/2 +output_seq_len, features)
        return X_des, X_det
    
    def call(self, input, training):
        X_des, X_det = self.prepare_input(input)
        enc_out = self.encoder(input, training)
        dec_out = self.decoder((X_des, X_det, enc_out), training)
        out = dec_out[:, -self.O:, :]
        return out
