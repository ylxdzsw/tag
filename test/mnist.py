def model_fn():
    x = tf.placeholder(tf.float32, shape=(None, 28, 28))
    y = tf.placeholder(tf.float32, shape=(None, 10,))
    x = tf.reshape(x, (-1, 28*28))
    hidden = tf.contrib.slim.fully_connected(x, 1024, activation_fn=tf.nn.sigmoid)
    output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=None)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    acc = tf.reduce_mean(tf.nn.softmax(output) * y)
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(tf.reduce_sum(loss))
    return optimizer

import time
import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from utils import info

BATCHSIZE=8

devices = (
    "GPU:0",
    "GPU:1",
)

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

with open("model.pb", "w") as fo:
    fo.write(pbtf.MessageToString(gdef))

import tge

x = [0, 3, 111, 114]
g = (tge.TGE(gdef, devices)
    .set_strategy({ node.name: [4 if i in x else 1, 1, 1] for i, node in enumerate(gdef.node) })
    .compile()
    .get_result()
)

with open("modified.pb", "w") as fo:
    fo.write(pbtf.MessageToString(g))

tf.reset_default_graph()
tf.import_graph_def(g)
graph = tf.get_default_graph()

x_tensor = graph.get_tensor_by_name("import/Placeholder/replica_0:0")
y_tensor = graph.get_tensor_by_name("import/Placeholder_1/replica_0:0")
opt = graph.get_operation_by_name("import/GradientDescent/replica_0")
init = graph.get_operation_by_name("import/init/replica_0")
acc_tensor = 10 * (
    graph.get_tensor_by_name("import/Mean/replica_0:0") +
    graph.get_tensor_by_name("import/Mean/replica_1:0")) / 2

sess = tf.Session()
sess.run(init)

def onehot(x):
    max = x.max() + 1
    return np.eye(max)[x]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = onehot(y_train.reshape(-1))
y_test = onehot(y_test.reshape(-1))

p = time.perf_counter()
for batch_id in range(10000):
    i = batch_id % int(60000 / BATCHSIZE)

    sess.run(opt, {
        x_tensor: x_train[BATCHSIZE*i:BATCHSIZE*(i+1)],
        y_tensor: y_train[BATCHSIZE*i:BATCHSIZE*(i+1)]
    })

    if i % 50 == 0:
        q = time.perf_counter()
        a = sess.run(acc_tensor, { x_tensor: x_test, y_tensor: y_test })
        info("acc:", a, "time:", (q - p) / 50)
        p = time.perf_counter()

