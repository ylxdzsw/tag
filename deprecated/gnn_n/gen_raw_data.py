def vgg(bsize=None):
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def resnet(bsize=None):
    from tensorflow.contrib.slim.nets import resnet_v2
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = resnet_v2.resnet_v2_101(x, 1000)
    output = tf.contrib.slim.flatten(output)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def mlp(bsize=None):
    x = tf.placeholder(tf.float32, shape=(bsize, 1024))
    y = tf.placeholder(tf.float32, shape=(bsize, 10,))
    hidden = tf.contrib.slim.fully_connected(x, 256, activation_fn=tf.nn.softmax)
    output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=tf.nn.softmax)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def lenet(bsize=None):
    slim = tf.contrib.slim
    x = tf.placeholder(tf.float32, shape=(bsize, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    net = slim.conv2d(x, 32, [5, 5])
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.conv2d(net, 64, [5, 5])
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.sigmoid)
    net = slim.fully_connected(net, 1000, activation_fn=None)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=net)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(tf.reduce_sum(loss))
    return optimizer

def inception(bsize=None):
    from tensorflow.contrib.slim.nets import inception
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = inception.inception_v3(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def transformer(bsize=None):
    import sys
    sys.path.insert(0, './transformer/')
    import transformer as transf
    from data import DatasetManager
    dm = DatasetManager("wmt14")
    dm.maybe_download_data_files()
    dm.load_vocab()
    transformer = transf.Transformer(
        num_heads=8,
        d_model=512,
        d_ff=2048,
        model_name="transformer",
        tf_sess_config=dict(allow_soft_placement=True)
    )
    train_params = dict(
        learning_rate=1e-4,
        batch_size=bsize,
        seq_len=10,
        max_steps=300000,
    )
    transformer.build_model("wmt14", dm.source_id2word, dm.target_id2word, 0,**train_params)
    loss = transformer._loss

    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def mobilenet(bsize=None):
    from tensorflow.contrib.slim.nets import mobilenet_v2
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = mobilenet_v2.mobilenet(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def nasnet(bsize=None):
    from tensorflow.contrib.slim.nets import nasnet
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = nasnet.build_nasnet_cifar(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def bert(bsize=None):
    import sys
    sys.path.insert(0, './bert/')
    from bert.runsquad import new_model_fn_builder
    import modeling
    bert_config = modeling.BertConfig.from_json_file("bert/bert_large/bert_config.json")
    model = new_model_fn_builder(bert_config)
    features = {}
    features["input_ids"]= tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,128)),tf.int32)
    features["input_mask"] = tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,128)),tf.int32)
    features["segment_ids"]=tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,128)),tf.int32)
    features["start_positions"] = tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,)),tf.int32)
    features["end_positions"] =tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,)),tf.int32)
    loss = model(features)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def berts(bsize=None):
    import sys
    sys.path.insert(0, './bert/')
    from bert.runsquad import new_model_fn_builder
    import modeling
    bert_config = modeling.BertConfig.from_json_file("bert/bert_small/bert_config.json")
    model = new_model_fn_builder(bert_config)
    features = {}
    features["input_ids"]= tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,128)),tf.int32)
    features["input_mask"] = tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,128)),tf.int32)
    features["segment_ids"]=tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,128)),tf.int32)
    features["start_positions"] = tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,)),tf.int32)
    features["end_positions"] =tf.cast(100*tf.placeholder(tf.float32,shape=(bsize,)),tf.int32)
    loss = model(features)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

# https://github.com/pangolulu/char-rnnlm-tensorflow/blob/master/model.py
def rnnlm2x(bsize=None):
    x = tf.placeholder(tf.int64, [bsize, 20])
    y = tf.placeholder(tf.float32, [bsize, 20, 256])
    embedding = tf.get_variable('embedding', [256, 16])
    x = tf.nn.embedding_lookup(embedding, x)
    x = tf.keras.layers.LSTM(4, return_sequences=True, unroll=True)(x)
    x = tf.keras.layers.LSTM(4, return_sequences=True, unroll=True)(x)
    x = tf.keras.layers.Dense(256, activation=None)(x)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def rnnlm4x(bsize=None):
    x = tf.placeholder(tf.int64, [bsize, 20])
    y = tf.placeholder(tf.float32, [bsize, 20, 256])
    embedding = tf.get_variable('embedding', [256, 16])
    x = tf.nn.embedding_lookup(embedding, x)
    x = tf.keras.layers.LSTM(4, return_sequences=True, unroll=True)(x)
    x = tf.keras.layers.LSTM(4, return_sequences=True, unroll=True)(x)
    x = tf.keras.layers.LSTM(4, return_sequences=True, unroll=True)(x)
    x = tf.keras.layers.LSTM(4, return_sequences=True, unroll=True)(x)
    x = tf.keras.layers.Dense(256, activation=None)(x)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(tf.reduce_sum(loss))
    return optimizer

import tensorflow as tf
import numpy as np
import pickle
from profiler import Profiler
from utils import adapt_batchsize

BATCHSIZE = {
    "vgg": 60,
    "resnet": 60,
    "mlp": 60,
    "lenet": 60,
    "inception": 60,
    "transformer": 600,
    "mobilenet": 60,
    "nasnet": 60,
    "bert": 2,
    "berts": 2,
    "rnnlm2x": 60,
    "rnnlm4x": 60
}

times = 5

import sys
model_fn = eval(sys.argv[1])
gtype = sys.argv[2]
batch_size = BATCHSIZE[sys.argv[1]]

import pathlib
pathlib.Path("raw_data/{}/{}".format(model_fn.__name__, gtype)).mkdir(parents=True, exist_ok=True)

# first dump the model
opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

with open("raw_data/{}/model.pickle".format(model_fn.__name__), 'wb') as f:
    pickle.dump(gdef, f)

for nrep in (1, 2, 3, 4, 6, 10):
    if batch_size % nrep != 0:
        continue

    prof_dict = {}
    tf.reset_default_graph()
    opt = model_fn()
    init = tf.global_variables_initializer()
    gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
    ps = [ Profiler(gdef, batch_size // nrep, sinks=["Adam"]) for _ in range(times) ]
    for node in gdef.node:
        prof_dict[node.name] = int(np.median([ p.profile(node.name, "/GPU:0") for p in ps ]))
    with open("raw_data/{}/{}/{}.pickle".format(model_fn.__name__, gtype, batch_size // nrep), 'wb') as f:
        pickle.dump(prof_dict, f)
