import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')


def PositionalEncoding(input):
    batch_size = tf.shape(input)[0].numpy()
    seq_len = tf.shape(input)[1].numpy()
    d_model = tf.shape(input)[2].numpy()
    pe = np.zeros((seq_len, d_model))  # [input_seq_len, d_model]
        
    position = np.arange(0, seq_len).reshape(-1, 1)  # [input_seq_len, 1]
    i2 = np.arange(0, d_model, 2)  # (0 2 4 ...)
    div_term = np.exp(i2 * np.log(10000.0) / d_model)  # [d_model/2, ]
    
    pe[:, 0::2] = np.sin(position / div_term)
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(position / div_term)
    else:
        pe[:, 1::2] = np.cos(position / div_term[:-1])
        pe[:, -1] = np.sin(position[:, 0] / div_term[-1])
    
    pe = pe[tf.newaxis, ...]  # [1, input_seq_len, d_model]
    pe = tf.tile(pe, [batch_size, 1, 1])  # [batch, seq_len, d_model]

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
    
    def call(self, query, key, value, training, masked_mha=False):  # 3x [batch, heads, seq_len, d_model / heads]
        QK = tf.matmul(query, key, transpose_b=True)  # [batch, heads, seq_len, seq_len]
        QK = tf.map_fn(lambda x: x/np.sqrt(self.d_k), QK)  # [batch, heads, seq_len, seq_len]
        if masked_mha:
            seq_len = QK.shape[-1]
            mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float64), -1, 0)
            mask = mask[tf.newaxis, tf.newaxis, :, :]
            QK += (mask * -1e19)
        QK = tf.nn.softmax(QK, axis=-1)  # [batch, heads, seq_len, seq_len]
        QKV = tf.matmul(QK, value)  # [batch, heads, seq_len, d_model / heads]
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
        
    def call(self, query, key, value, training): # (Q, K, V) : 3x [batch, seq_len, d_model]
        Q, K, V = self.createQKV(query, key, value)  # [batch, heads, seq_len, d_model / heads]
        scaled_dot_attention = self.dot_product(Q, K, V, training, self.masked_mha)  # [batch, heads, seq_len, d_model / heads]
        tr_sda = tf.transpose(scaled_dot_attention, perm=[0, 2, 1, 3])  # [batch, seq_len, heads, d_model / heads]
        concatenated = tf.reshape(tr_sda, (-1, tr_sda.shape[1], tr_sda.shape[2] * tr_sda.shape[3]))  # [batch, seq_len, d_model]
        res = self.out(concatenated)
        return res  # [batch, seq_len, d_model]

    def createQKV(self, query, key, value):  # 3x [batch, seq_len, d_model]
        q_list = [self.query(query) for _ in range(self.heads)]  # [(batch, seq_len, d_k), ...]
        k_list = [self.key(key) for _ in range(self.heads)]
        v_list = [self.value(key) for _ in range(self.heads)]
        Q_4d = tf.stack(q_list, axis=0)  # [heads, seq_len, batch, d_k]
        K_4d = tf.stack(k_list, axis=0)
        V_4d = tf.stack(v_list, axis=0)
        result_Q = tf.transpose(Q_4d, perm=[1, 0, 2, 3])  # [batch, heads, seq_len, d_k]
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

    def call(self, input, training):  # [batch, seq_len, d_model]
        x = self.multihead_attn(input, input, input, training)  # [batch, seq_len, d_model]
        x = self.dropout(x, training=training)
        x += input
        x = self.normalize1(x)  # [batch, seq_len, d_model]
        
        y = self.feed_forward(x)
        y += x
        y = self.normalize2(y)  # [batch, seq_len, d_model]
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
        self.multihead_attn_heads = config['multihead_attn_heads']
        self.d_k = config['d_k']
        self.d_v = config['d_v']
        self.d_ff = config['d_ff']
        self.dropout_rate = config['dropout_rate']
        self.encoder_layers_num = config['encoder_layers']
        self.encoder_layers = [EncoderLayer(config) for _ in range(config['encoder_layers'])]
        self.embed = tf.keras.layers.Dense(config['d_model'], activation='linear')
            
    def call(self, input, training):  # [batch, seq_len, features]
        x = self.embed(input)  # [batch, seq_len, d_model]
        x = PositionalEncoding(x)  # [batch, seq_len, d_model]
        for i in range(self.encoder_layers_num):
            x = self.encoder_layers[i](x, training)  # [batch, seq_len, d_model]
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
        
    def call(self, decoder_output, encoder_output, training):  # [batch, current_seq_len, d_model), [batch, seq_len, d_model]
        # first sublayer --- masked multi head attention for decoder output
        x = self.masked_multihead_attn(decoder_output, decoder_output, decoder_output, training)  # [batch, current_seq_len, d_model]
        x = self.dropout1(x, training=training)
        x += decoder_output
        x = self.normalize1(x)  # [batch, current_seq_len, d_model]
        
        # second sublayer --- multi head attention for encoder output (key, value) and decoder output (query)
        y = self.multihead_attn(x, encoder_output, encoder_output, training)  # [batch, current_seq_len, d_model]
        y = self.dropout2(y, training=training)
        y += x
        y = self.normalize2(y)  # [batch, current_seq_len, d_model]
        
        # third sublayer --- feed forward
        z = self.feed_forward(y)  # [batch, current_seq_len, d_model]
        z = self.dropout3(z)
        z += y
        z = self.normalize3(z)  # [batch, current_seq_len, d_model]
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
        self.embed = tf.keras.layers.Dense(config['d_model'], activation='linear')
        self.generated_sequence = list()
        self.last_known_data = None
        self.decoder_layers_num = config['decoder_layers']
        self.decoder_layers = [DecoderLayer(config) for _ in range(config['decoder_layers'])]

        self.linear_out = tf.keras.layers.Dense(config['d_out'], activation='linear')
        #self.linear_out = tf.keras.layers.Dense(config['d_out'],activation='linear',bias_initializer='zeros')  # for (stationary) data with mean 0

    def call_train(self, encoder_output, target, training):  # [batch, seq_Len, d_model] [batch, output_seq_len, target_features]
        shifted_inputs = tf.concat([self.last_known_data, target], axis=1)  # [batch, seq_len, target_features]
        shifted_inputs = self.embed(shifted_inputs)  # [batch, seq_len, d_model]
        shifted_inputs = PositionalEncoding(shifted_inputs)
        for i in range(self.decoder_layers_num):
            shifted_inputs = self.decoder_layers[i](shifted_inputs, encoder_output, training)  # [batch, seq_len, d_model]
        out = self.linear_out(shifted_inputs)  # [batch, seq_len, target_features]
        return out[:, :-1, :]

    def call_inference(self, encoder_output, training):
        self.generated_sequence.append(self.last_known_data)
        # decoder input in first step is last data point from input sequence
        # with every step the decoder input longer by 1
        for i in range(self.output_seq_len):
            new_input = tf.concat(self.generated_sequence, axis=1)  # [batch, current_seq_len, target_features]
            decoder_output_emb = self.embed(new_input)  # [batch, current_seq_len, d_model]
            decoder_output_emb = PositionalEncoding(decoder_output_emb)
            for j in range(self.decoder_layers_num):
                decoder_output_emb = self.decoder_layers[j](decoder_output_emb, encoder_output, training)  # [batch, current_seq_len, d_model]
            out = self.linear_out(decoder_output_emb)  # [batch, current_seq_len, target_features]
            self.generated_sequence.append(out[:, -1:, :])
        
        out = self.generated_sequence
        out = tf.concat(out, axis=1)  # [batch, target_seq_len+1, target_features]
        # discard first element of the sequence which is the last known data point therefore not part of the forecast
        out = out[:, 1:, :]  # [batch, target_seq_len, target_features]
        self.generated_sequence = []
        return out
    
    def call(self, last_data, encoder_output, target, training):
        self.last_known_data = last_data
        if training == True:
            out = self.call_train(encoder_output, target, training)
            return out
        else:
            out = self.call_inference(encoder_output, training)
            return out


class Transformer(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(Transformer, self).__init__()
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
        self.batch_size = config['batch_size']
        self.target_idx = config['target_idx']

    def call(self, input, target, training):
        enc_out = self.encoder(input, training)
        dec_out = self.decoder(input[:, -1:, self.target_idx:self.target_idx+1], enc_out, target, training)
        out = tf.concat(dec_out, axis=1)
        return out
    