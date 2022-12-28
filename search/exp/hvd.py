def vgg(bsize=None):
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2)
    return x, y, loss, optimizer

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()

config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

bsize = 96 // 2

x, y, loss, opt = vgg(bsize)
opt = hvd.DistributedOptimizer(opt)
train_op = opt.minimize(tf.reduce_sum(loss))

hooks = [hvd.BroadcastGlobalVariablesHook(0)]

tf.train.get_or_create_global_step()

sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

import time
last_time = 0
for i in range(20):
    sess.run(train_op, feed_dict={
        x: np.random.rand(bsize, 224, 224, 3),
        y: np.zeros((bsize, 1000))
    })
    new_time = time.time()
    print(new_time - last_time)
    last_time = new_time

# with tf.train.MonitoredTrainingSession(checkpoint_dir=('/tmp/train_logs' if hvd.rank() == 0 else None),
#                                        config=config,
#                                        hooks=hooks) as mon_sess:
#   while not mon_sess.should_stop():
#     # Perform synchronous training.
#     mon_sess.run(train_op, feed_dict={
#         x: tf.convert_to_tensor(np.random.rand(96, 224, 244, 3)),
#         y: tf.convert_to_tensor(np.zeros(96, 1000))
# })
