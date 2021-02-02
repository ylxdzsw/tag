import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample, evaluate, base_strategies, score
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
    model = Model()

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
            record['reference'] = []
            for nodemask, ncclmask, psmask in base_strategies(record):
                loss_env = score(*evaluate(record, nodemask, ncclmask, psmask))
                record['reference'].append((loss_env, nodemask, ncclmask, psmask))
            save(records, "records")

        if 'elites' not in record:
            best = record['reference'][0]
            for loss_env, nodemask, ncclmask, psmask in record['reference']:
                if loss_env < best[0]:
                    best = loss_env, nodemask, ncclmask, psmask
            record['elites'] = [best]

        op_feats     = tf.convert_to_tensor(record["op_feats"], dtype=tf.float32)
        task_feats   = tf.convert_to_tensor(record["task_feats"], dtype=tf.float32)
        tensor_feats = tf.convert_to_tensor(record["tensor_feats"], dtype=tf.float32)
        link_feats   = tf.convert_to_tensor(record["link_feats"], dtype=tf.float32)
        place_feats  = tf.convert_to_tensor(record["place_feats"], dtype=tf.float32)
        model.set_graph(record["graph"])

        # search
        if epoch > 200 and epoch % 20 == 0:
            logit = model([op_feats, task_feats, tensor_feats, link_feats, place_feats], training=True)

            placement = sample(logit)
            loss_env, nodemask, ncclmask, psmask = search(record, placement)

            if loss_env < record['elites'][-1][0]:
                record['elites'].append((loss_env, nodemask, ncclmask, psmask))
                record["elites"] = record["elites"][-4:]

            info(record_id, loss_env, [ x for x, *_ in record['reference'] ])

        # learn
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logit = model([op_feats, task_feats, tensor_feats, link_feats, place_feats], training=True)

            loss = 0
            for loss_env, nodemask, ncclmask, psmask in record['elites']:
                nodemask = np.sign(nodemask)
                loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(nodemask.astype(np.float32), logit))
            loss /= len(record['elites'])

            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)

            info(record_id, loss.numpy())

            grads = tape.gradient(loss, model.trainable_weights)
            # info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # checkpoint
        if epoch % 20 == 0:
            info("==== save ====")
            model.save_weights('weights')
            save(records, "records")
