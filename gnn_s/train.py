import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample, evaluate, sample_and_evaluate, reference
from search import search
from utils import save, load, info

try:
    records = load("records")
    info("load saved records")
except:
    records = get_all_data()
    info("no saved records")
    save(records, "records")

with tf.device("/gpu:0"):
    model = Model(records[0]["op_table"])

    try:
        model.load_weights('weights')
        info("load saved weight")
    except:
        info("no saved weight")

    optimizer = tf.keras.optimizers.Adam(learning_rate=.0001, clipnorm=6)
    L2_regularization_factor = .00001

    for epoch in range(20000):
        record_id = np.random.randint(len(records))
        record = records[record_id]

        if 'reference' not in record:
            reference(record)
            save(records, "records")

        if 'elites' not in record:
            nodep = np.ones((len(record['cgroups']) * len(record['devices']), 3), dtype=np.float) / 3
            ncclp = np.ones(len(record['cgroups']), dtype=np.float) / 2
            record['elites'] = [search(record, nodep, ncclp, n_gen=25)]
            save(records, "records")

        cnfeats = tf.convert_to_tensor(record["cnfeats"], dtype=tf.float32)
        cefeats = tf.convert_to_tensor(record["cefeats"], dtype=tf.float32)
        cntypes = tf.convert_to_tensor(record["cntypes"], dtype=tf.float32)
        tnfeats = tf.convert_to_tensor(record["tnfeats"], dtype=tf.float32)
        tefeats = tf.convert_to_tensor(record["tefeats"], dtype=tf.float32)
        model.set_graphs(record["cgraph"], record["tgraph"])
        model.set_groups(record["cgroups"], record["tgroups"])

        # search
        if epoch % 10 == 0:
            nodelogit, nccllogit = model([cnfeats, cefeats, cntypes, tnfeats, tefeats], training=False)
            nodep = tf.nn.softmax(nodelogit).numpy()
            ncclp = tf.math.sigmoid(nccllogit).numpy()
            loss_env, nodemask, ncclmask = search(record, nodep, ncclp)
            record['elites'].append((loss_env, nodemask, ncclmask))
            record["elites"] = record["elites"][-100:]

            info(record_id, loss_env, record['reference'])

        # learn
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            nodelogit, nccllogit = model([cnfeats, cefeats, cntypes, tnfeats, tefeats], training=True)

            # info(tf.nn.softmax(nodelogit).numpy())
            # info(nodelogit.numpy())

            loss = 0
            for loss_env, nodemask, ncclmask in record['elites']:
                nodemask = tf.one_hot(nodemask, depth=3).numpy()
                loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(nodemask.astype(np.float32), nodelogit))
                loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ncclmask.astype(np.float32), nccllogit))

            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)

            info(loss.numpy())

            grads = tape.gradient(loss, model.trainable_weights)
            # info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # checkpoint
        if epoch % 50 == 0:
            info("==== save ====")
            model.save_weights('weights')
            save(records, "records")
