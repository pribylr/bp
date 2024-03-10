import tensorflow as tf
import numpy as np

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
    def __init__(self, d_k, d_v, **kwargs):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v   
    
    def call(self, inputs, to_mask=False):  # (Q, K, V) : (batch, heads, seq_len, features+2 / heads)
        QK = tf.matmul(inputs[0], inputs[1], transpose_b=True)  # (batch, heads, seq_len, seq_len)
        QK = tf.map_fn(lambda x: x/np.sqrt(self.d_k), QK)  # (batch, heads, seq_len, seq_len)
        if to_mask:
            seq_len = QK.shape[-1]
            mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = mask[tf.newaxis, tf.newaxis, :, :]
            QK += (mask * -1e19)
        QK = tf.nn.softmax(QK, axis=-1)  # (batch, heads, seq_len, seq_len)
        QKV = tf.matmul(QK, inputs[2])  # (batch, heads, seq_len, features+2 / heads)
        return QKV
        
    def get_config(self):
        config = super(ScaledDotProductAttention, self).get_config()
        config.update({
            'd_k': self.d_k,
            'd_v': self.d_v
        })
        return config

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, heads, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads
        self.dot_product = ScaledDotProductAttention(d_k, d_v)
        
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
            input_shape[0][-1],  # features+2
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform')
        
    def call(self, inputs): # (Q, K, V) : x.shape == (batch, seq_len, features+2)
        Q, K, V = self.createQKV(inputs)  # (batch, heads, seq_len, features+2 / heads)
        scaled_dot_attention = self.dot_product((Q, K, V), False)  # (batch, heads, seq_len, features+2 / heads)
        tr_sda = tf.transpose(scaled_dot_attention, perm=[0, 2, 1, 3])
        concatenated = tf.reshape(tr_sda, (tr_sda.shape[0], tr_sda.shape[1], tr_sda.shape[2] * tr_sda.shape[3]))  # (batch, seq_len, features+2)
        res = self.linear_out(concatenated)
        return res  # (batch, seq_len, features+2)

    def createQKV(self, input):  # (x, x, x) : x.shape == (batch, seq_len, features+2)
        q_list = [self.query(input[0]) for _ in range(self.heads)]  # [(batch, seq_len, d_k), ...]
        k_list = [self.key(input[1]) for _ in range(self.heads)]
        v_list = [self.value(input[2]) for _ in range(self.heads)]
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
    def __init__(self, d_k: int, d_v: int, multi_heads: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(EncoderLayer, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.multi_heads = multi_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.multihead_attn = MultiHeadAttention(d_k, d_v, multi_heads)
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.attn_normalize = tf.keras.layers.LayerNormalization()

        self.ff_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ff_normalize = tf.keras.layers.LayerNormalization()
        
    def build(self, input_shape):
        self.ff_conv1D_1 = tf.keras.layers.Conv1D(filters=self.d_ff, kernel_size=1, activation='relu')
        self.ff_conv1D_2 = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)

    def call(self, input):  # (batch, seq_len, features+2)
        x = self.multihead_attn((input, input, input))  # (batch, seq_len, features+2)
        x = self.attn_dropout(x)
        x = self.attn_normalize(x + input)  # (batch, seq_len, features+2)

        y = self.ff_conv1D_1(x)  # (batch, seq_len, d_ff)
        y = self.ff_conv1D_2(y)  # (batch, seq_len, features+2)
        y = self.ff_dropout(y)
        y = self.ff_normalize(x + y)  # (batch, seq_len, features+2)
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
    def __init__(self, input_seq_len: int, multi_heads: int, d_ff: int, encoder_layers: int, dropout_rate: float = 0.1, **kwargs):
        super(Encoder, self).__init__()
        self.input_seq_len = input_seq_len
        self.multi_heads = multi_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.encoder_layers = encoder_layers
        self.encoder_layers_list = list()
        self.time_embedding = Time2Vector(input_seq_len)    
            
    def call(self, input):
        x = self.time_embedding(input)  # (batch, seq_len, 2)
        x = tf.keras.layers.Concatenate(axis=-1)([input, x])  # (batch, seq_len, features+2)
        
        d_k = x.shape[-1] / self.multi_heads
        d_v = d_k
        enc_layer = EncoderLayer(d_k, d_v, self.multi_heads, self.d_ff, self.dropout_rate)
        out = enc_layer(x)
        for _ in range(self.encoder_layers-1):
            out = enc_layer(out)
        return out

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'input_seq_len': self.input_seq_len,
            'multi_heads': self.multi_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'encoder_layers': self.encoder_layers,
        })
        return config

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_k: int, d_v: int, multi_heads: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(DecoderLayer, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.multi_heads = multi_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        #self.

    #def call(self, input):
        
        

