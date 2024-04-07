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
                

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(EncoderLayer, self).__init__()
        self.autocorrelation = Autocorrelation(config)
        self.series_decomp1 = Series_decomp(config)
        self.series_decomp2 = Series_decomp(config)
        self.feed_forward = FeedForward(config, config['d_model'])
    
    def call(self, input, training):  # (batch, seq_len, d_model)
        #print('enc layer AC start')
        x = self.autocorrelation((input, input, input))  # (batch, seq_len, d_model)
        #print('enc layer AC end')
        x += input
        
        #print('enc layer series decomp 1 start')
        S1, _ = self.series_decomp1(x)
        #print('enc layer series decomp 1 end')

        #print('enc layer FF start')
        x = self.feed_forward(S1, training)
        #print('enc layer FF end')
        x += S1

        #print('enc layer series decomp 2 start')
        S2, _ = self.series_decomp2(x)
        #print('enc layer series decomp 2 end')
        return S2  # (batch, seq_len, d_model)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__()
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
        #print('decoder layer AC 1 start')
        x = self.autocorrelation1((input[0], input[0], input[0]))  # (batch, seq_len/2 + O, d_model)
        #print('decoder layer AC 1 end')
        x += input[0]
        #print('decoder layer series decomp 1 start')
        S1, T1 = self.series_decomp1(x)
        #print('decoder layer series decomp 1 end')
        #print('decoder layer AC 2 start')
        y = self.autocorrelation2((S1, input[2], input[2]))  # (batch, seq_len/2 + O, d_model)
        #print('decoder layer AC 2 end')
        y += S1
        #print('decoder layer series decomp 2 start')
        S2, T2 = self.series_decomp2(y)
        #print('decoder layer series decomp 2 end')
        
        #print('decoder layer FF start')
        z = self.feed_forward(S2, training)
        #print('decoder layer FF end')
        z += S2
        #print('decoder layer series decomp 3 start')
        S3, T3 = self.series_decomp3(z)
        #print('decoder layer series decomp 3 start')
        T = input[1]
        #print('decoder layer T mlp start')
        T += self.mlp1(T1, training)
        T += self.mlp2(T2, training)
        T += self.mlp3(T3, training)
        #print('decoder layer T mlp end')
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
