import tensorflow as tf
from dgl.nn.tensorflow import edge_softmax, GATConv

def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention 乘上value
    output = tf.matmul(attention_weights, v) # （.., seq_len_v, depth）

    return output, attention_weights

class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights




def point_wise_feed_forward_network(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, ddf, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MutilHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, ddf)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        # 多头注意力网络
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2



class GAT(tf.keras.Model):
    def __init__(self, computation_feature_length,device_num,max_replica_num_per_device):
        super(GAT, self).__init__()

        num_hidden = 256
        num_heads = 8
        GAT_options = (0.5, 0.5, 0.2) # feat_drop_rate, attn_drop_rate, negative_slope
        num_rnn_hidden = 256

        self.device_num = device_num
        self.max_replica_num_per_device = max_replica_num_per_device

        self.computation_gat_layers = [
            GATConv(computation_feature_length, num_hidden, num_heads, *GAT_options, False, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, True, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, True, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, True, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, 1, *GAT_options, False, None)
        ]

        #self.device_gat_layers = [
        #    GATConv(device_feature_length, num_hidden, num_heads, *GAT_options, False, tf.nn.elu),
        #    GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, True, tf.nn.elu),
        #    GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, True, tf.nn.elu),
        #    GATConv(num_hidden * num_heads, num_hidden, 1, *GAT_options, False, None)
        #]

        self.rnn_layers = [
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True))
        ]


        self.encode_layer = [EncoderLayer(num_hidden, num_heads, num_hidden, 0.1)
                            for _ in range(12)]

        self.pre_final = tf.keras.layers.Dense(device_num*(max_replica_num_per_device+1)+2, activation=None) # put 0, 1, 2 replicas
        self.device_final = [tf.keras.layers.Dense(max_replica_num_per_device+1, activation=tf.nn.log_softmax) for i in range(device_num)] # put 0, 1, 2 replicas
        self.ps_final = tf.keras.layers.Dense(2, activation=tf.nn.log_softmax) # put 0, 1, 2 replicas

    def set_graphs(self, computation_graph,init_group):
        self.computation_graph = computation_graph
        self.init_group = init_group

    def call(self, computation_features):


        x = computation_features
        for layer in self.computation_gat_layers:
            x = layer(self.computation_graph, x)
            x = tf.reshape(x, (x.shape[0], -1))
        computation_embedding = x
        group_embedding = tf.math.unsorted_segment_max(computation_embedding, self.init_group, tf.reduce_max(self.init_group) + 1)
        x = tf.reshape(group_embedding, [1,group_embedding.shape[0], -1])

        for layer in self.encode_layer:
            x = layer(x,True,None)
        x = tf.reshape(x, [group_embedding.shape[0],-1])
        x = self.pre_final(x)
        devices = [x[:,i*(self.max_replica_num_per_device+1):(i+1)*(self.max_replica_num_per_device+1)] for i in range(self.device_num)]
        ps = x[:,-2:]
        devices = [self.device_final[i](item) for i,item in enumerate(devices)]
        ps = self.ps_final(ps)
        result = tf.concat([*devices,ps],axis = 1)
        #x = self.final(x) # contrary to what has been stated in documentation, the Dense layer is applied on the last axis of input tensor (https://github.com/tensorflow/tensorflow/issues/30882)
        return result
