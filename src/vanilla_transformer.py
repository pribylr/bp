import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')


def PositionalEncoding(input):
    batch_size = tf.shape(input)[0].numpy()
    seq_len = tf.shape(input)[1].numpy()
    d_model = tf.shape(input)[2].numpy()
    pe = np.zeros((seq_len, d_model))  # (input_seq_len, d_model)
        
    position = np.arange(0, seq_len).reshape(-1, 1)  # (input_seq_len, 1)        
    i2 = np.arange(0, d_model, 2)  # [0 2 ...]        
    div_term = np.exp(i2 * np.log(10000.0) / d_model)  # (d_model / 2, ) ()

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
        
    pe = pe[tf.newaxis, ...]  # (1, input_seq_len, d_model) 
    pe = tf.tile(pe, [batch_size, 1, 1])  # (batch, seq_len, d_model)

    input += pe
    return input


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


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
    
    def call(self, query, key, value, training, masked_mha=False):  # (Q, K, V) : (batch, heads, seq_len, features+2 / heads)
        QK = tf.matmul(query, key, transpose_b=True)  # (batch, heads, seq_len, seq_len)
        QK = tf.map_fn(lambda x: x/np.sqrt(self.d_k), QK)  # (batch, heads, seq_len, seq_len)
        if masked_mha and training:
            seq_len = QK.shape[-1]
            mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float64), -1, 0)
            mask = mask[tf.newaxis, tf.newaxis, :, :]
            QK += (mask * -1e19)
        QK = tf.nn.softmax(QK, axis=-1)  # (batch, heads, seq_len, seq_len)
        QKV = tf.matmul(QK, value)  # (batch, heads, seq_len, features+2 / heads)
        return QKV
        
    def get_config(self):
        config = super(ScaledDotProductAttention, self).get_config()
        config.update({
            'd_k': self.d_k,
            'd_v': self.d_v
        })
        return config


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, config, masked_mha=False, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.heads = config['multihead_attn_heads']
        self.masked_mha = masked_mha
        self.d_model = config['d_model']
        self.dot_product = ScaledDotProductAttention(config)
        
        self.query = tf.keras.layers.Dense(self.d_k, activation='linear')
        self.key = tf.keras.layers.Dense(self.d_k, activation='linear')
        self.value = tf.keras.layers.Dense(self.d_v, activation='linear')
        self.out = tf.keras.layers.Dense(self.d_model, activation='linear')
        
    def call(self, query, key, value, training): # (Q, K, V) : x.shape == (batch, seq_len, features+2)
        Q, K, V = self.createQKV(query, key, value)  # (batch, heads, seq_len, features+2 / heads)
        scaled_dot_attention = self.dot_product(Q, K, V, training, self.masked_mha)  # (batch, heads, seq_len, features+2 / heads)
        tr_sda = tf.transpose(scaled_dot_attention, perm=[0, 2, 1, 3])  # (batch, seq_len, heads, features+2 / heads)
        concatenated = tf.reshape(tr_sda, (-1, tr_sda.shape[1], tr_sda.shape[2] * tr_sda.shape[3]))  # (batch, seq_len, features+2)
        res = self.out(concatenated)
        return res  # (batch, seq_len, features+2)

    def createQKV(self, query, key, value):  # (x, x, x) : x.shape == (batch, seq_len, features+2)
        q_list = [self.query(query) for _ in range(self.heads)]  # [(batch, seq_len, d_k), ...]
        k_list = [self.key(key) for _ in range(self.heads)]
        v_list = [self.value(key) for _ in range(self.heads)]
        Q_4d = tf.stack(q_list, axis=0)  # (heads, seq_len, batch, d_k)
        K_4d = tf.stack(k_list, axis=0)
        V_4d = tf.stack(v_list, axis=0)
        result_Q = tf.transpose(Q_4d, perm=[1, 0, 2, 3])  # (batch, heads, seq_len, d_k)
        result_K = tf.transpose(K_4d, perm=[1, 0, 2, 3])
        result_V = tf.transpose(V_4d, perm=[1, 0, 2, 3])
        return result_Q, result_K, result_V
        
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'd_k': self.d_k,
            'd_v': self.d_v,
            'heads': self.heads
        })
        return config


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(EncoderLayer, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.multihead_attn_heads = config['multihead_attn_heads']
        self.d_ff = config['d_ff']
        self.dropout_rate = config['dropout_rate']
        self.multihead_attn = MultiHeadAttention(config, masked_mha=False)
        
        self.dropout = tf.keras.layers.Dropout(config['dropout_rate'])
        self.normalize1 = tf.keras.layers.LayerNormalization()

        self.feed_forward = FeedForward(config, config['d_model'])
        self.normalize2 = tf.keras.layers.LayerNormalization()

    def call(self, input, training):  # (batch, seq_len, features+2)
        x = self.multihead_attn(input, input, input, training)  # (batch, seq_len, features+2)
        x = self.dropout(x, training=training)
        x += input
        x = self.normalize1(x)  # (batch, seq_len, features+2)

        y = self.feed_forward(x)
        y += x
        y = self.normalize2(y)  # (batch, seq_len, features+2)
        return y

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_k': self.d_k,
            'd_v': self.d_v,
            'multi_heads': self.multi_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        })
        return config


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__()
        self.input_seq_len = config['input_seq_len']
        self.multihead_attn_heads = config['multihead_attn_heads']
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.d_ff = config['d_ff']
        self.dropout_rate = config['dropout_rate']
        self.encoder_layers_num = config['encoder_layers']
        self.encoder_layers = [EncoderLayer(config) for _ in range(config['encoder_layers'])]
        self.embed = tf.keras.layers.Dense(config['d_model'], activation='linear')
            
    def call(self, input, training):
        x = self.embed(input)
        for i in range(self.encoder_layers_num):
            x = self.encoder_layers[i](x, training)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(DecoderLayer, self).__init__()
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.multihead_attn_heads = config['multihead_attn_heads']
        self.d_ff = config['d_ff']
        self.dropout_rate = config['dropout_rate']
        
        self.masked_multihead_attn = MultiHeadAttention(config, masked_mha=True)
        self.multihead_attn = MultiHeadAttention(config, masked_mha=False)
        self.dropout1 = tf.keras.layers.Dropout(config['dropout_rate'])
        self.normalize1 = tf.keras.layers.LayerNormalization()

        self.dropout2 = tf.keras.layers.Dropout(config['dropout_rate'])
        self.normalize2 = tf.keras.layers.LayerNormalization()

        self.dropout3 = tf.keras.layers.Dropout(config['dropout_rate'])
        self.normalize3 = tf.keras.layers.LayerNormalization()
        
        self.feed_forward = FeedForward(config, config['d_model'])
        
    def call(self, decoder_output, encoder_output, training):  # (batch, seq_len, features+2), (batch, seq_len, features+2)
        # first sublayer --- masked multi head attention for decoder output
        x = self.masked_multihead_attn(decoder_output, decoder_output, decoder_output, training)  # (batch, seq_len, features+2)
        x = self.dropout1(x, training=training)
        x += decoder_output
        x = self.normalize1(x)  # (batch, seq_len, features+2)
        
        # second sublayer --- multi head attention for encoder output (key, value) and decoder output (query)
        y = self.multihead_attn(x, encoder_output, encoder_output, training)  # q, k, v
        y = self.dropout2(y, training=training)
        y += x
        y = self.normalize2(y)
        
        # third sublayer --- feed forward
        z = self.feed_forward(y)
        z = self.dropout3(z)
        z += y
        z = self.normalize3(z)  # (batch, seq_len, features+2)
        return z


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__()
        self.output_seq_len = config['output_seq_len']
        self.multihead_attn_heads = config['multihead_attn_heads']
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.d_ff = config['d_ff']
        self.d_out = config['d_out']
        self.dropout_rate = config['dropout_rate']
        self.d_model = config['d_model']

        # self.pe_list = [PositionalEncoding(config, 1)]
        # for i in range(config['output_seq_len']):
        #     self.pe_list.append(PositionalEncoding(config, i+1))
        self.pe_list = [PositionalEncoding(config, i+1) for i in range(config['output_seq_len'])]
    
        self.generated_sequence = list()
        self.last_known_data = None

        self.decoder_layers_num = config['decoder_layers']
        self.decoder_layers = [DecoderLayer(config) for _ in range(config['decoder_layers'])]

        self.linear_out = tf.keras.layers.Dense(config['d_out'], activation='linear')
        #self.linear_out = tf.keras.layers.Dense(config['d_out'],activation='linear',bias_initializer='zeros')  # for stationary data with mean 0

    def call(self, transformer_input, encoder_output, training):
        # decoder input in first step is last data point from input sequence
        # with every step the decoder input longer by 1
        self.generated_sequence.append(transformer_input[:, -1:, :])
        for i in range(self.output_seq_len):
            new_input = tf.concat(self.generated_sequence, axis=1)
            decoder_output_emb = self.pe_list[i](new_input, training)
            for j in range(self.decoder_layers_num):
                decoder_output_emb = self.decoder_layers[j](decoder_output_emb, encoder_output, training)
            out = self.linear_out(decoder_output_emb)
            self.generated_sequence.append(out[:, -1:, :])
        
        out = self.generated_sequence
        out = tf.concat(out, axis=1)
        out = out[:, 1:, :]
        self.generated_sequence = []
        return out


class Transformer(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(Transformer, self).__init__()
        self.input_seq_len = config['input_seq_len']
        self.output_seq_len = config['output_seq_len']
        self.multihead_attn_heads = config['multihead_attn_heads']
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.d_ff = config['d_ff']
        self.d_out = config['d_out']
        self.encoder_layers = config['encoder_layers']
        self.decoder_layers = config['decoder_layers']
        self.dropout_rate = config['dropout_rate']
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def call(self, input, training):
        enc_out = self.encoder(input, training)
        dec_out = self.decoder(input, enc_out, training)
        out = tf.concat(dec_out, axis=1)
        return out
    