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
            GATConv(node_hidden, activation=tf.nn.sigmoid) #tf.identity)
        ]

        self.final_node = tf.keras.layers.Dense(1, activation=None)
        self.final_nccl = tf.keras.layers.Dense(1, activation=None)
        self.condition = tf.keras.layers.Dense(node_hidden, activation=tf.nn.sigmoid)
        self.final_place = tf.keras.layers.Dense(1, activation=None)
        self.final_ps = tf.keras.layers.Dense(1, activation=None)
        self.final_critic = tf.keras.layers.Dense(1, activation=None)

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

        node_logit = tf.squeeze(self.final_node(op_feats), axis=1)
        node_sampled = int(tf.random.categorical(tf.reshape(node_logit, (1, -1)), 1).numpy()[0, 0])
        node_sampled_feat = op_feats[node_sampled:node_sampled+1, :]

        device_feats = tf.concat([device_feats, tf.repeat(node_sampled_feat, device_feats.shape[0], axis=0)], axis=1)
        device_feats = self.condition(device_feats)

        nccl_logit = tf.squeeze(self.final_nccl(node_sampled_feat), axis=1)
        place_logit = tf.squeeze(self.final_place(device_feats), axis=1)
        ps_logit = tf.squeeze(self.final_ps(device_feats), axis=1)

        critic = tf.squeeze(self.final_critic(tf.reduce_mean(op_feats, axis=0, keepdims=True)), axis=1)

        return node_logit, node_sampled, nccl_logit, place_logit, ps_logit, critic

def run_episode(record, initial_state, model, max_steps=8):
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action_entropys = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    target_score = initial_state[3]
    state = initial_state

    for t in range(max_steps):
        action, value = run_model(record, model, state)
        action_sampled, action_prob, action_entropy = sample_action(action)
        # info(action_sampled, action_prob, action_entropy)
        values = values.write(t, value)
        action_probs = action_probs.write(t, action_prob)
        action_entropys = action_entropys.write(t, action_entropy)
        state, reward = env_step(record, state, target_score, action_sampled)
        info(record['id'], *action_sampled, value.numpy()[0], reward)
        rewards = rewards.write(t, reward)

    if np.random.rand() < .05:
        if len(record['seeds']) < 100:
            record['seeds'].append(state)
        else:
            record['seeds'][np.random.randint(len(record['seeds']))] = state

    return action_probs.stack(), action_entropys.stack(), values.stack(), rewards.stack()

def sample_action(action):
    node_logit, node_sampled, nccl_logit, place_logit, ps_logit = action

    nccl_sampled = sample_logits(nccl_logit)[0]
    place_sampled = sample_logits(place_logit)
    ps_sampled = tf.random.categorical(tf.reshape(ps_logit, (1, -1)), 1).numpy()[0, 0]

    prob_node = tf.nn.softmax(node_logit)[node_sampled]
    entropy_node = -tf.reduce_sum(tf.nn.log_softmax(node_logit) * tf.nn.softmax(node_logit))

    prob_ps = tf.nn.softmax(ps_logit)[ps_sampled]
    entropy_ps = -tf.reduce_sum(tf.nn.log_softmax(ps_logit) * tf.nn.softmax(ps_logit))

    ppnccl = tf.math.sigmoid(nccl_logit)
    prob_nccl = (1 - nccl_sampled) * (1 - ppnccl) + nccl_sampled * ppnccl
    entropy_nccl = -(ppnccl * tf.math.log(ppnccl) + (1 - ppnccl) * tf.math.log(1 - ppnccl))

    ppplace = tf.math.sigmoid(place_logit)
    prob_place = tf.reduce_prod((1 - place_sampled) * (1 - ppplace) + place_sampled * ppplace)
    entropy_place = -tf.reduce_sum(ppplace * tf.math.log(ppplace) + (1 - ppplace) * tf.math.log(1 - ppplace))

    samples = node_sampled, nccl_sampled, place_sampled, ps_sampled
    prob = prob_node * prob_ps * prob_nccl * prob_place
    entropy = entropy_node + entropy_ps + entropy_nccl + entropy_place

    return samples, prob, entropy

def run_model(record, model, state):
    nodemask, ncclmask, psmask, score, feedback = state
    features = prepare_features(record, nodemask, ncclmask, psmask, feedback)
    node_logit, node_sampled, nccl_logit, place_logit, ps_logit, critic = model(features)
    return (node_logit, node_sampled, nccl_logit, place_logit, ps_logit), critic

def env_step(record, state, target_score, action):
    nodemask, ncclmask, psmask, original_score, feedback = state
    node_sampled, nccl_sampled, place_sampled, ps_sampled = action
    nodemask_new = np.copy(nodemask)
    nodemask_new[node_sampled, :] = place_sampled
    ncclmask_new = np.copy(ncclmask)
    ncclmask_new[node_sampled] = nccl_sampled
    psmask_new = np.copy(psmask)
    psmask_new[node_sampled] = ps_sampled
    score, feedback = evaluate_with_feedback(record, nodemask_new, ncclmask_new, psmask_new)
    # reward = int(score < target_score)
    return (nodemask_new, ncclmask_new, psmask_new, score, feedback), -score

def sample_logits(logit, e=.00):
    p = tf.math.sigmoid(logit)
    def f(x):
        if np.random.rand() < e:
            return np.random.choice(2)
        else:
            return int(np.random.rand() < x)
    return np.vectorize(f)(p)

eps = np.finfo(np.float32).eps.item()

def get_expected_return(rewards, gamma=.95, standardize=True):
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)

    return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(record, action_probs, action_entropy, values, returns):
    advantage = returns - values
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    critic_loss = huber_loss(values, returns)
    entropy_loss = -tf.math.reduce_sum(action_entropy)
    # info(actor_loss, critic_loss, entropy_loss)
    if record['nvisit'] < 20:
        return 0.01*actor_loss + 2*critic_loss + 0.01*entropy_loss
    else:
        return 0.02*actor_loss + 2*critic_loss + 0.005*entropy_loss

def train_step(record, model, optimizer):
    initial_state = record['seeds'][np.random.randint(len(record['seeds']))]
    with tf.GradientTape() as tape:
        action_prob, action_entropy, values, rewards = run_episode(record, initial_state, model)
        returns = get_expected_return(rewards, standardize=False)
        action_prob, values, returns = [tf.expand_dims(x, 1) for x in [action_prob, values, returns]]
        loss = compute_loss(record, action_prob, action_entropy, values, returns)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    episode_reward = tf.math.reduce_sum(rewards).numpy()
    return episode_reward

def prepare_features(record, nodemask, ncclmask, psmask, feedback):
    CL2, Bmax = record['scaler']

    op_feedbacks = []
    for node in record['gdef'].node:
        # if node.name not in feedback['op_makespan'] or node.name not in feedback['op_idle_after']:
            # info(node.name)

        op_feedbacks.append([
            feedback['op_makespan'].get(node.name, 0) / CL2, # the ratio of makespan and computation time?
            feedback['op_idle_after'].get(node.name, 0) / CL2,
        ])

    op_feats = np.hstack((
        record["op_feats"],
        # np.reshape(ncclmask, (-1, 1)), # TODO: expand if grouped
        np.reshape(feedback['leftout'], (-1, 1)),
        op_feedbacks
    ))
    device_feats = np.hstack((
        record["device_feats"],
        np.reshape(feedback['device_total_utilization'], (-1, 1)),
        np.array([[feedback['device_peak_memory'][i] / record['topo_spec'].tasks[record['devices'][i][1]].memory] for i in range(len(record['devices'])) ]),
    ))
    tensor_feats = record["tensor_feats"]
    link_feats = record["link_feats"]

    place_extra = []
    for g_id, group in enumerate(record['groups']):
        for op_id in group:
            for dev_id in range(len(record['devices'])):
                place_extra.append([ nodemask[g_id, dev_id], int(psmask[g_id] == dev_id) ])

    place_feats = np.hstack((
        record["place_feats"],
        np.array(place_extra),
    ))

    return op_feats, device_feats, tensor_feats, link_feats, place_feats
