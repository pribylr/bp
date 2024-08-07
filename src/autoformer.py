import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')

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
        self.fc1 = tf.keras.layers.Dense(config['d_ff'], activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(config['dropout_rate'])
        self.fc2 = tf.keras.layers.Dense(d_out, activation='linear')
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
        self.d_model = config['d_model']
        self.heads = config['ac_heads']
        self.c = config['c']
        self.input_seq_len = config['input_seq_len']
        self.query = tf.keras.layers.Dense(config['d_model']/config['ac_heads'], activation='linear')
        self.key = tf.keras.layers.Dense(config['d_model']/config['ac_heads'], activation='linear')
        self.value = tf.keras.layers.Dense(config['d_model']/config['ac_heads'], activation='linear')
        self.out = tf.keras.layers.Dense(config['d_model'], activation='linear')
        self.k = int(config['c']*np.log(config['input_seq_len']))

    def createQKV(self, input):
        Q = input[0]
        K = input[1]
        V = input[2]
        if Q.shape[1] > K.shape[1]:  # Q longer seq_len -- zero filling
            paddings = tf.constant([[0, 0], [0, Q.shape[1] - K.shape[1]], [0, 0]])
            K = tf.pad(K, paddings)
            V = tf.pad(V, paddings)
        if Q.shape[1] < K.shape[1]:  # Q shorter seq_len -- truncation
            len = Q.shape[1]
            K = K[:, -len:, :]
            V = V[:, -len:, :]
            
        queries = [self.query(Q) for _ in range(self.heads)]  # [(batch, seq_len, d_model/heads), ...]
        keys = [self.query(K) for _ in range(self.heads)]  # [(batch, seq_len, d_model/heads), ...]
        values = [self.query(V) for _ in range(self.heads)]  # [(batch, seq_len, d_model/heads), ...]
        Q_4d = tf.stack(queries, axis=1)  # (batch, heads, seq_len, d_model/heads)
        K_4d = tf.stack(keys, axis=1)  # (batch, seq_len, heads, d_model/heads)
        V_4d = tf.stack(values, axis=1)  # (batch, seq_len, heads, d_model/heads)
        Q_4d = tf.cast(Q_4d, tf.complex64)
        K_4d = tf.cast(K_4d, tf.complex64)
        
        Q_4d = tf.transpose(Q_4d, perm=[0, 1, 3, 2])  # (batch, heads, d_model/heads, seq_len)
        K_4d = tf.transpose(K_4d, perm=[0, 1, 3, 2])
        V_4d = tf.transpose(V_4d, perm=[0, 1, 3, 2])
        return Q_4d, K_4d, V_4d    
    
    def fast_fourier_operations(self, input):
        # along the seq_len dimension
        Q = tf.signal.fft(input[0])
        K = tf.signal.fft(input[1])
        QK = tf.multiply(Q, tf.math.conj(K))
        corr = tf.signal.ifft(QK)
        corr = tf.math.real(corr)
        corr = tf.reduce_mean(corr, axis=[0, 1, 2])  # (seq_len )
        corr = tf.reshape(corr, shape=(1, -1))  # (1, seq_len)
        return corr
    
    def time_delay_train(self, W_topk, I_topk, V):  # (1, k) (1, k) (batch, heads, d_model/heads, seq_len)
        B, H, D, L = V.shape
        rolled = []
        for i in range(self.k):
            shift = tf.keras.backend.eval(I_topk[:, i]).item()
            rolled.append(tf.roll(V, shift=-shift, axis=-1))
        Vs_rolled = tf.stack(rolled, axis=1)  # (batch, k, heads, d_model/heads, seq_len)
        W_topk = tf.cast(tf.reshape(W_topk, [1, -1, 1, 1, 1]), tf.float64)  # (1, k, 1, 1, 1)
        Vs_weighted = Vs_rolled*W_topk  # (batch, k, heads, d_model/heads, seq_len)
        R = tf.reduce_sum(Vs_weighted, axis=1)  # (batch, heads, d_model/heads, seq_len)
        R = tf.reshape(R, [B, L, H*D])  # (batch, seq_len, d_model)
        return R

    def time_delay_infer(self, W_topk, I_topk, V):
        B, H, D, L = V.shape
        V = tf.concat([V, V], axis=-1)  # (batch, heads, d_model/heads, 2seq_len)
        index = tf.range(L)
        index = tf.reshape(index, [1, 1, 1, L])
        index = tf.tile(index, [B, H, D, 1])
        R = tf.zeros_like(V[:, :, :, :L])

        for i in range(self.k):
            tmp = tf.expand_dims(tf.expand_dims(tf.expand_dims(I_topk[:, i], -1), -1), -1)
            tmp = tf.tile(tmp, [B, H, D, L])
            W_topk_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(W_topk[:, i], -1), -1), -1)
            W_topk_exp = tf.tile(W_topk_exp, [B, H, D, L])
            W_topk_exp = tf.cast(W_topk_exp, tf.float64)
            tmp += index
            indices_shape = tf.shape(tmp)
            batch_indices = tf.range(0, B)
            heads_indices = tf.range(0, H)
            d_model_indices = tf.range(0, D)
            B_tmp, H_tmp, D_tmp, L_tmp = tf.meshgrid(batch_indices, heads_indices, d_model_indices, tf.range(0, tf.shape(tmp)[-1]), indexing='ij')
            gather_indices = tf.stack([B_tmp, H_tmp, D_tmp, tmp], axis=-1)
            
            result_tensor = tf.gather_nd(V, gather_indices)  # (batch, heads, d_model/heads, seq_len)
            R += result_tensor * W_topk_exp
        
        R = tf.transpose(R, perm=[0, 3, 1, 2])
        R = tf.reshape(R, [B, L, H*D])  # (batch, seq_len, d_model)
        return R
    
    def call(self, input, training):
        Q, K, V = self.createQKV(input)  # (batch, heads, d_model/heads, seq_len)
        corr = self.fast_fourier_operations((Q, K))
        W_topk, I_topk = tf.math.top_k(corr, k=self.k)
        W_topk = tf.nn.softmax(W_topk)
        if training:
            res = self.time_delay_train(W_topk, I_topk, V)
            res = self.out(res)
            return res
        else:
            res = self.time_delay_infer(W_topk, I_topk, V)
            res = self.out(res)
            return res
            

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(EncoderLayer, self).__init__()
        self.autocorrelation = Autocorrelation(config)
        self.series_decomp1 = Series_decomp(config)
        self.series_decomp2 = Series_decomp(config)
        self.feed_forward = FeedForward(config, config['d_model'])
    
    def call(self, input, training):  # [batch, seq_len, d_model]
        x = self.autocorrelation((input, input, input))  # [batch, seq_len, d_model]
        x += input
                
        S1, _ = self.series_decomp1(x)  # [batch, seq_len, d_model]
        x = self.feed_forward(S1, training)  # [batch, seq_len, d_model]
        x += S1

        S2, _ = self.series_decomp2(x)
        return S2  # [batch, seq_len, d_model]


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__()
        self.encoder_layers_num = config['encoder_layers']
        self.encoder_layers = [EncoderLayer(config) for _ in range(config['encoder_layers'])]
        self.embed = tf.keras.layers.Dense(config['d_model'])

    def call(self, input, training):  # [batch_size, input_seq_lne, features]
        x = self.embed(input)  # [batch, seq_len, d_model]
        for i in range(self.encoder_layers_num):
            x = self.encoder_layers[i](x, training)  # [batch, seq_len, d_model]
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
        self.mlp1 = FeedForward(config, config['d_model'])
        self.mlp2 = FeedForward(config, config['d_model'])
        self.mlp3 = FeedForward(config, config['d_model'])

    def call(self, input, training):
        # [0] -- embedded S  [batch, seq_len/2 + O, d_model]
        # [1] -- T           [batch, seq_len/2 + O, features]
        # [2] -- enc out     [batch, seq_len, d_model]
        x = self.autocorrelation1((input[0], input[0], input[0]))  # [batch, seq_len/2 + O, d_model]
        x += input[0]
        S1, T1 = self.series_decomp1(x)  # 2x [batch, seq_len/2 + O, d_model]

        y = self.autocorrelation2((S1, input[2], input[2]))  # [batch, seq_len/2 + O, d_model]
        y += S1
        S2, T2 = self.series_decomp2(y)  # 2x [batch, seq_len/2 + O, d_model]
        
        z = self.feed_forward(S2, training)  # [batch, seq_len/2 + O, d_model]
        z += S2
        S3, T3 = self.series_decomp3(z)  # 2x [batch, seq_len/2 + O, d_model]

        T = input[1]
        
        tmp = self.mlp1(T1, training) 
        T += tmp
        #T += self.mlp1(T1, training)
        T += self.mlp2(T2, training)
        T += self.mlp3(T3, training)
        return S3, T  # [batch, seq_len/2 + O, d_model], [batch, seq_len/2 + O, d_data]


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__()
        self.decoder_layers_num = config['decoder_layers']
        self.decoder_layers = [DecoderLayer(config) for _ in range(config['decoder_layers'])]
        self.embed_S = tf.keras.layers.Dense(config['d_model'])
        self.embed_T = tf.keras.layers.Dense(config['d_model'])
        self.mlp = FeedForward(config, config['d_out'])
        self.linear_out = tf.keras.layers.Dense(config['d_out'], activation='linear')
        #self.linear_out = tf.keras.layers.Dense(config['d_out'], activation='linear', bias_initializer='zeros')
    
    def call(self, input, training):  # X_des (batch, seq_len/2 + O, features), X_det (batch, seq_len/2 + O, features), enc_out (batch, seq_len, d_model)
        S = self.embed_S(input[0])  # [batch, seq_len/2 + O, d_model]
        T = self.embed_T(input[1])  # [batch, seq_len/2 + O, features]
        for i in range(self.decoder_layers_num):
            S, T = self.decoder_layers[i]((S, T, input[2]), training)  # 2x [batch, seq_len/2 + O, d_model]
        
        S = self.mlp(S, training)
        out = self.linear_out(S+T)
        return out


class Autoformer(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(Autoformer, self).__init__()
        self.input_seq_len = config['input_seq_len']
        self.O = config['O']
        self.pool_size = config['pool_size']
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.series_decomp = Series_decomp(config)
        
    def prepare_input(self, input):  # (batch, seq_len, features)
        X_ens, X_ent = self.series_decomp(input[:, (input.shape[1])//2:, :])  # (batch, seq_len/2, features)
        mean = tf.reduce_mean(input, axis=1, keepdims=True)
        mean = tf.tile(mean, [1, self.O, 1])  # (batch, output_seq_len, features)
        zeros = tf.zeros([tf.shape(input)[0], self.O, input.shape[2]], dtype=tf.float64)  # (batch, output_seq_len, features)
        
        X_det = tf.concat([X_ent, mean], axis=1)  # (batch, seq_len/2 +output_seq_len, features)
        X_des = tf.concat([X_ens, zeros], axis=1)  # (batch, seq_len/2 +output_seq_len, features)
        return X_des, X_det
    
    def call(self, input, training):  # [batch_size, input_seq_lne, features]
        X_des, X_det = self.prepare_input(input)
        enc_out = self.encoder(input, training)
        dec_out = self.decoder((X_des, X_det, enc_out), training)
        out = dec_out[:, -self.O:, :]
        return out
