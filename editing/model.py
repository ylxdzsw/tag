import tensorflow as tf
import dgl.function as fn
import numpy as np
from utils import info, positional_encoding

all_etypes = ["link", "prev", "succ", "place", "serve"]

class GConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self, out_feats, activation=None):
        super(GConv, self).__init__()
        self.activation = activation
        self.layers = { etype: tf.keras.layers.Dense(out_feats, activation=None) for etype in all_etypes }

    def call(self, graph, op_feats, device_feats, edge_feats):
        op_dst, device_dst = [], []
        for stype, etype, dtype in graph.canonical_etypes:
            g = graph[etype].local_var()

            if stype == 'op':
                g.srcdata['i'] = op_feats
            elif stype == 'device':
                g.srcdata['i'] = device_feats

            g.apply_edges(fn.copy_u('i', 's'))
            edata = tf.concat([g.edata.pop('s'), edge_feats[etype]], axis=1)
            g.edata['e'] = self.layers[etype](edata)
            g.update_all(fn.copy_e('e', 'm'), fn.mean(msg='m', out='o'))

            if dtype == 'op':
                op_dst.append(g.dstdata['o'])
            elif dtype == 'device':
                device_dst.append(g.dstdata['o'])

        op_dst = tf.math.add_n(op_dst) / len(op_dst)
        device_dst = tf.math.add_n(device_dst) / len(device_dst)

        # return self.activation(op_feats + op_dst), self.activation(device_feats + device_dst)
        return tf.concat([op_feats, self.activation(op_dst)], axis=1), tf.concat([device_feats, self.activation(device_dst)], axis=1)
        # return self.activation(op_dst), self.activation(device_dst)

class GATConv(tf.keras.layers.Layer):
    '''Graph Attention layer that concats the edge features before sending message'''
    def __init__(self, out_feats, n_heads=4, dropout=.0, activation=None, batch_norm=True):
        super(GATConv, self).__init__()
        self.convs = { etype: GATConvSlice(out_feats, n_heads, dropout, batch_norm) for etype in all_etypes }
        self.op_fc = tf.keras.layers.Dense(out_feats, activation=activation)
        self.device_fc = tf.keras.layers.Dense(out_feats, activation=activation)

    def call(self, graph, op_feats, device_feats, edge_feats):
        op_dst, device_dst = [], []
        for stype, etype, dtype in graph.canonical_etypes:
            if stype == 'op':
                srcfeats = op_feats
            elif stype == 'device':
                srcfeats = device_feats

            efeats = edge_feats[etype]

            if dtype == 'op':
                dstfeats = op_feats
            elif dtype == 'device':
                dstfeats = device_feats

            out = self.convs[etype](graph[etype], srcfeats, efeats, dstfeats)

            if dtype == 'op':
                op_dst.append(out)
            elif dtype == 'device':
                device_dst.append(out)

        op_dst = tf.math.add_n(op_dst) / len(op_dst)
        device_dst = tf.math.add_n(device_dst) / len(device_dst)

        # batch norm here?
        op_dst = self.op_fc(op_dst)
        device_dst = self.device_fc(device_dst)

        return tf.concat([op_feats, op_dst], axis=1), tf.concat([device_feats, device_dst], axis=1)

from dgl.nn.tensorflow.softmax import edge_softmax
class GATConvSlice(tf.keras.layers.Layer):
    def __init__(self, out_feats, n_heads, dropout, batch_norm, negative_slope=0.2):
        super(GATConvSlice, self).__init__()
        self.n_heads = n_heads
        self.out_feats = out_feats

        xinit = tf.keras.initializers.VarianceScaling(scale=np.sqrt(2), mode="fan_avg", distribution="untruncated_normal")
        self.fc_src = tf.keras.layers.Dense(out_feats * n_heads, use_bias=False, kernel_initializer=xinit)
        self.fc_dst = tf.keras.layers.Dense(out_feats * n_heads, use_bias=False, kernel_initializer=xinit)
        self.attn_l = tf.Variable(initial_value=xinit(shape=(1, n_heads, out_feats), dtype='float32'), trainable=True)
        self.attn_r = tf.Variable(initial_value=xinit(shape=(1, n_heads, out_feats), dtype='float32'), trainable=True)
        self.src_batch_norm = tf.keras.layers.BatchNormalization() if batch_norm else None
        self.dst_batch_norm = tf.keras.layers.BatchNormalization() if batch_norm else None
        self.attn_drop = tf.keras.layers.Dropout(rate=dropout)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=negative_slope)

    def call(self, graph, srcfeats, efeats, dstfeats):
        g = graph.local_var()

        g.srcdata['i'] = srcfeats
        g.apply_edges(fn.copy_u('i', 's'))
        edata = tf.concat([g.edata.pop('s'), efeats], axis=1)

        srcfeats = self.fc_src(edata)
        dstfeats = self.fc_dst(dstfeats)

        srcfeats = self.src_batch_norm(srcfeats) if self.src_batch_norm is not None else srcfeats
        dstfeats = self.dst_batch_norm(dstfeats) if self.dst_batch_norm is not None else dstfeats

        srcfeats = tf.reshape(srcfeats, (-1, self.n_heads, self.out_feats))
        dstfeats = tf.reshape(dstfeats, (-1, self.n_heads, self.out_feats))

        el = tf.reduce_sum(srcfeats * self.attn_l, axis=-1, keepdims=True)
        er = tf.reduce_sum(dstfeats * self.attn_r, axis=-1, keepdims=True)

        g.edata.update({'el': el})
        g.dstdata.update({'er': er})
        g.apply_edges(fn.e_add_v('el', 'er', 'e'))
        e = self.leaky_relu(g.edata.pop('e'))
        g.edata['x'] = srcfeats * self.attn_drop(edge_softmax(g, e))
        g.update_all(fn.copy_e('x', 'm'), fn.sum('m', 'o'))
        return tf.reshape(g.dstdata['o'], (-1, self.n_heads * self.out_feats))

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        node_hidden = 1024
        edge_hidden = 64

        self.op_trans = tf.keras.layers.Dense(node_hidden, activation=tf.nn.sigmoid)
        self.op_normalizer = tf.keras.layers.BatchNormalization()
        self.device_trans = tf.keras.layers.Dense(node_hidden, activation=tf.nn.sigmoid)
        self.device_normalizer = tf.keras.layers.BatchNormalization()
        self.edge_trans = { etype: tf.keras.layers.Dense(edge_hidden, activation=tf.nn.sigmoid) for etype in all_etypes }
        self.edge_normalizers = { etype: tf.keras.layers.BatchNormalization() for etype in all_etypes }

        self.gconv_layers = [
            GATConv(node_hidden, activation=tf.nn.sigmoid),
            GATConv(node_hidden, activation=tf.nn.sigmoid),
            GATConv(node_hidden, activation=tf.nn.sigmoid),
            GATConv(node_hidden, activation=tf.nn.sigmoid) #tf.identity)
        ]

        self.final_place = tf.keras.layers.Dense(1, activation=None)
        self.final_nccl = tf.keras.layers.Dense(1, activation=None)

    def set_graph(self, graph, segments):
        # self.graph = graph.to('gpu:0')
        self.graph = graph
        self.segments = segments

    def call(self, inputs):
        [op_feats, device_feats, tensor_feats, link_feats, place_feats] = inputs

        op_feats = self.op_trans(op_feats)
        op_feats = tf.math.unsorted_segment_mean(op_feats, *self.segments['op'])
        op_feats = self.op_normalizer(op_feats)
        device_feats = self.device_trans(device_feats)
        device_feats = self.device_normalizer(device_feats)

        edge_feats = {
            "link": link_feats,
            "prev": tensor_feats,
            "succ": tensor_feats,
            "place": place_feats,
            "serve": place_feats
        }
        for etype in all_etypes:
            x = self.edge_trans[etype](edge_feats[etype])
            if etype in self.segments:
                x = tf.math.unsorted_segment_mean(x, *self.segments[etype])
            edge_feats[etype] = self.edge_normalizers[etype](x)

        for gconv_layer in self.gconv_layers:
            op_feats, device_feats = gconv_layer(self.graph, op_feats, device_feats, edge_feats)

        g = self.graph['place'].local_var()
        g.srcdata['i'] = op_feats
        g.dstdata['i'] = device_feats
        g.edata['i'] = edge_feats['place']
        g.apply_edges(lambda edge: { 'd': self.final_place(tf.concat([edge.src['i'], edge.data['i'], edge.dst['i']], axis=1)) })
        node_logit = tf.reshape(tf.squeeze(g.edata['d'], axis=1), (op_feats.shape[0], device_feats.shape[0]))
        nccl_logit = tf.squeeze(self.final_nccl(op_feats), axis=1)

        return node_logit, nccl_logit
