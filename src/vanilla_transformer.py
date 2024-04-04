import tensorflow as tf
import numpy as np


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


class Time2Vector(tf.keras.layers.Layer):
    def __init__(self, seq_len: int, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len
        
    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weights_linear', shape=(int(self.seq_len),), initializer='uniform', trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear', shape=(int(self.seq_len),), initializer='uniform', trainable=True)
        
        self.weights_periodic = self.add_weight(name='weights_periodic', shape=(int(self.seq_len),), initializer='uniform',trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic', shape=(int(self.seq_len),), initializer='uniform', trainable=True)

    def call(self, input): # (batch, seq_len, features)
        x = tf.math.reduce_mean(input, axis=-1) # (batch, seq_len)
        time_linear = self.weights_linear * x + self.bias_linear# (batch, seq_len)
        time_linear = tf.expand_dims(time_linear, axis=-1) # (batch, seq_len, 1)
        
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)  # (batch, seq_len)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)  # (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1)  # (batch size, seq_len, 2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'seq_len': self.seq_len
        })
        return config


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
            mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
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
        self.dot_product = ScaledDotProductAttention(config)
        
    def build(self, input_shape):
        self.query = tf.keras.layers.Dense(
            self.d_k, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='glorot_uniform')
        self.key = tf.keras.layers.Dense(
            self.d_k,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
        self.value = tf.keras.layers.Dense(
            self.d_v,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
        self.linear_out = tf.keras.layers.Dense(
            input_shape[-1],  # features+2
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
        
    def call(self, query, key, value, training): # (Q, K, V) : x.shape == (batch, seq_len, features+2)
        Q, K, V = self.createQKV(query, key, value)  # (batch, heads, seq_len, features+2 / heads)
        scaled_dot_attention = self.dot_product(Q, K, V, training, self.masked_mha)  # (batch, heads, seq_len, features+2 / heads)
        tr_sda = tf.transpose(scaled_dot_attention, perm=[0, 2, 1, 3])  # (batch, seq_len, heads, features+2 / heads)
        concatenated = tf.reshape(tr_sda, (-1, tr_sda.shape[1], tr_sda.shape[2] * tr_sda.shape[3]))  # (batch, seq_len, features+2)
        res = self.linear_out(concatenated)
        return res  # (batch, seq_len, features+2)

    def createQKV(self, query, key, value):  # (x, x, x) : x.shape == (batch, seq_len, features+2)
        q_list = [self.query(query) for _ in range(self.heads)]  # [(batch, seq_len, d_k), ...]
        k_list = [self.key(key) for _ in range(self.heads)]
        v_list = [self.value(key) for _ in range(self.heads)]
        q_tr_list = [tf.transpose(tensor, perm=[1, 0, 2]) for tensor in q_list]  # [(seq_len, batch, d_k), ...]
        k_tr_list = [tf.transpose(tensor, perm=[1, 0, 2]) for tensor in k_list]
        v_tr_list = [tf.transpose(tensor, perm=[1, 0, 2]) for tensor in v_list]
        Q_4d = tf.stack(q_tr_list, axis=0)  # (heads, seq_len, batch, d_k)
        K_4d = tf.stack(k_tr_list, axis=0)
        V_4d = tf.stack(v_tr_list, axis=0)
        result_Q = tf.transpose(Q_4d, perm=[2, 0, 1, 3])  # (batch, heads, seq_len, d_k)
        result_K = tf.transpose(K_4d, perm=[2, 0, 1, 3])
        result_V = tf.transpose(V_4d, perm=[2, 0, 1, 3])
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
        
        self.attn_dropout = tf.keras.layers.Dropout(config['dropout_rate'])
        self.attn_normalize = tf.keras.layers.LayerNormalization()

        self.ff_dropout = tf.keras.layers.Dropout(config['dropout_rate'])
        self.ff_normalize = tf.keras.layers.LayerNormalization()

        self.feed_forward = FeedForward(config, config['d_model'])


    def call(self, input, training):  # (batch, seq_len, features+2)
        x = self.multihead_attn(input, input, input, training)  # (batch, seq_len, features+2)
        x = self.attn_dropout(x)  # ?
        x += input
        #x = self.attn_normalize(x + input)  # (batch, seq_len, features+2)

        y = self.feed_forward(x)
        y += x
        #y = self.ff_normalize(x + y)  # (batch, seq_len, features+2)
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
        self.time_embedding = Time2Vector(config['input_seq_len'])
            
    def call(self, input, training):
        x = self.time_embedding(input)  # (batch, seq_len, 2)
        x = tf.keras.layers.Concatenate(axis=-1)([input, x])  # (batch, seq_len, features+2)
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
        self.attn_dropout1 = tf.keras.layers.Dropout(config['dropout_rate'])
        self.attn_normalize1 = tf.keras.layers.LayerNormalization()

        self.attn_dropout2 = tf.keras.layers.Dropout(config['dropout_rate'])
        self.attn_normalize2 = tf.keras.layers.LayerNormalization()

        self.ff_dropout = tf.keras.layers.Dropout(config['dropout_rate'])
        self.ff_normalize = tf.keras.layers.LayerNormalization()
        
        self.feed_forward = FeedForward(config, config['d_model'])
        
    def call(self, decoder_output, encoder_output, training):  # (batch, seq_len, features+2), (batch, seq_len, features+2)
        # first sublayer --- masked multi head attention for decoder output
        x = self.masked_multihead_attn(decoder_output, decoder_output, decoder_output, training)  # (batch, seq_len, features+2)
        x = self.attn_dropout1(x)
        x += decoder_output
        #x = self.attn_normalize1(x + decoder_output)  # (batch, seq_len, features+2)
        # second sublayer --- multi head attention for encoder output (key, value) and decoder output (query)
        y = self.multihead_attn(x, encoder_output, encoder_output, training)  # q, k, v
        y = self.attn_dropout2(y)
        y += x
        #y = self.attn_normalize2(y + x)
        #tf.print('decoder 2nd sublayer output:', y)
        # third sublayer --- feed forward
        z = self.feed_forward(y)
        z += y
        #z = self.ff_normalize(z + y)  # (batch, seq_len, features+2)
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

        self.t2v_list = [Time2Vector(1)]
        for i in range(config['output_seq_len']):
            self.t2v_list.append(Time2Vector(i+1))
    
        self.generated_sequence = list()
        self.last_known_data = None

        self.decoder_layers_num = config['decoder_layers']
        self.decoder_layers = [DecoderLayer(config) for _ in range(config['decoder_layers'])]

        self.linear_out = tf.keras.layers.Dense(
            config['d_out'],
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')

    def call(self, transformer_input, encoder_output, training):  # decoder_output == (), encoder_output == (batch, seq_len, features+2)
        self.last_known_data = transformer_input[:, -1:, :]
        # outter loop: generating sequence of length self.output_seq_len
        # inner loop: number of decoder layers

        # first outter loop
        time_emb = self.t2v_list[0](self.last_known_data)  # (batch, seq_len, 2)
        decoder_output_emb = tf.keras.layers.Concatenate(axis=-1)([self.last_known_data, time_emb])  # (batch, seq_len, features+2)
        
        for i in range(self.decoder_layers_num):
            decoder_output_emb = self.decoder_layers[i](decoder_output_emb, encoder_output, training)
        out = tf.cast(self.linear_out(decoder_output_emb), dtype=tf.float64)
        self.generated_sequence.append(out)

        # rest of outter loop
        for i in range(self.output_seq_len-1):
            new_input = tf.concat(self.generated_sequence, axis=1)
            time_emb = self.t2v_list[i+1](new_input)
            decoder_output_emb = tf.keras.layers.Concatenate(axis=-1)([new_input, time_emb])
            
            for j in range(self.decoder_layers_num):
                decoder_output_emb = self.decoder_layers[j](decoder_output_emb, encoder_output, training)
            out = tf.cast(self.linear_out(decoder_output_emb), dtype=tf.float64)
            self.generated_sequence.append(out[:, -1:, :])
        out = self.generated_sequence
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
    