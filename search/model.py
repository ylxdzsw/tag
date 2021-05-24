import tensorflow as tf
import dgl.function as fn
import numpy as np
from environment import evaluate_with_feedback
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

        return op_feats + op_dst, device_feats + device_dst
        # return tf.concat([op_feats, op_dst], axis=1), tf.concat([device_feats, device_dst], axis=1)

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

class GNN(tf.keras.Model):
    def __init__(self):
        super(GNN, self).__init__()

        node_hidden = 256
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
            GATConv(node_hidden, activation=None)
        ]

    def set_graph(self, graph):
        # self.graph = graph.to('gpu:0')
        self.graph = graph

    def call(self, inputs):
        [op_feats, device_feats, tensor_feats, link_feats, place_feats] = inputs

        op_feats = self.op_trans(op_feats)
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
            edge_feats[etype] = self.edge_normalizers[etype](x)

        for gconv_layer in self.gconv_layers:
            op_feats, device_feats = gconv_layer(self.graph, op_feats, device_feats, edge_feats)

        return op_feats, device_feats

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        hidden = 256

        self.per_device_linear = tf.keras.layers.Dense(node_hidden, activation=tf.nn.sigmoid)
        self.hidden = tf.keras.layers.Dense(node_hidden, activation=tf.nn.sigmoid)
        self.final_linear = tf.keras.layers.Dense(node_hidden, activation=None)

    def call(self, device_embeddings, op_embedding, placement_masks, communication_masks):
        all_logis = []
        for i in range(len(communication_masks)):
            x = self.per_device_linear(tf.concat([device_embeddings, placement_masks[i]], axis=1))
            x = tf.concat([tf.reduce_mean(x, axis=0, keepdims=true), op_embedding, communication_masks[i]], axis=1)
            x = tf.squeeze(self.final_linear(self.hidden(x)))
            all_logis.append(x)

        return tf.nn.log_softmax(tf.concat(all_logis, axis=0))

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.gnn = GNN()
        self.decoder = Decoder()

    def call(self, inputs, masks):
        embeddings = self.gnn(inputs)
        return self.decoder(*embeddings, *masks)

def encode_features_no_runtime(state):
    record = state.record
    return record['op_feats'], record['device_feats'], record['tensor_feats'], record['link_feats'], record['place_feats']

def encode_features(state):
    record, feedback = state.record, state.result[1]

    CL2, Bmax = record['scaler']

    op_feedbacks = []
    for group in record['op_groups']:
        op_makespan = 0 # TODO: use the overall makespan of the group, rather than the average of each op
        op_idle_after = 0
        for node_id in group:
            node = record['gdef'][node_id]
            # if node.name not in feedback['op_makespan'] or node.name not in feedback['op_idle_after']:
                # info(node.name)
            op_makespan += feedback['op_makespan'].get(node.name, 0) / CL2 # the ratio of makespan and computation time?
            op_idle_after += feedback['op_idle_after'].get(node.name, 0) / CL2

        op_feedbacks.append([
            op_makespan / len(group),
            op_idle_after / len(group),
        ])

    op_communication_strategy = [ state.get_action(gid).to_mask()[1] for gid in range(len(record['op_groups'])) ]

    op_feats = np.hstack((
        record["op_feats"],
        op_communication_strategy, # TODO: how to feed the sfb?
        op_feedbacks
    ))

    device_feats = np.hstack((
        record["device_feats"],
        [ [ max( feedback['device_peak_memory'][i] / record['topo_spec'].tasks[tid].memory for i in range(len(record['device_segments'])) if tid == record['device_segments'][i] ) ] for tid in range(record['topo_spec'].ntasks) ],
        [ [ np.average( feedback['device_total_utilization'][i] for i in range(len(record['device_segments'])) if tid == record['device_segments'][i] ) ] for tid in range(record['topo_spec'].ntasks) ]
    ))

    tensor_feats = record["tensor_feats"]

    link_feats = record["link_feats"]

    place_extra = []
    for gid in range(len(record['op_groups'])):
        placement = state.get_action(gid).to_mask()[0]
        for tid in range(record['topo_spec'].ntasks):
            place_extra.append([ placement[tid] ])

    place_feats = np.hstack((
        record["place_feats"],
        place_extra,
    ))

    return op_feats, device_feats, tensor_feats, link_feats, place_feats

def policy(model, state, actions):
    feats = encode_features(state)
    masks = zip(*(action.to_mask() for action in actions))
    log_p = model(feats, masks)
    return log_p
