import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample_logits, evaluate_with_feedback, score, invalidity, sample_and_evaluate_with_feedback
from utils import save, load, info
from baseline import gen_baselines, eval_baselines

from multiprocessing import Pool
pool = Pool(16)

try:
    records = load("records")
    info("load saved records")
except:
    records = get_all_data()
    info("no saved records")
    save(records, "records")

def prepare_features(record, nodemask, ncclmask, psmask, feedback):
    CL2, Bmax = record['scaler']

    op_feats = np.hstack((
        record["op_feats"],
        np.reshape(ncclmask, (-1, 1)),
        np.reshape(feedback['leftout'], (-1, 1))
    ))
    device_feats = np.hstack((
        record["device_feats"],
        np.reshape(feedback['device_total_utilization'], (-1, 1)),
        np.array([[feedback['device_peak_memory'][i] / record['topo_spec'].tasks[record['devices'][i][1]].memory] for i in range(len(record['devices'])) ]),
    ))
    tensor_feats = record["tensor_feats"]
    link_feats = record["link_feats"]

    place_extra = []
    for op_id in range(len(record['gdef'].node)):
        for dev_id in range(len(record['devices'])):
            place_extra.append([ nodemask[op_id, dev_id], int(psmask[op_id] == dev_id) ])

    place_feats = np.hstack((
        record["place_feats"],
        np.array(place_extra),
    ))

    return op_feats, device_feats, tensor_feats, link_feats, place_feats

with tf.device("/gpu:0"):
    model = Model()

    try:
        model.load_weights('weights')
        info("load saved weight")
    except:
        info("no saved weight")

    optimizer = tf.keras.optimizers.Adam(learning_rate=.0001, clipnorm=6)
    L2_regularization_factor = .00001
    similarity_factor = .1

    for epoch in range(20000):
        # record_id = np.random.randint(len(records))
        record = records[0]

        if 'baselines' not in record:
            baselines = gen_baselines(record)
            eval_baselines(record, baselines)
            record['baselines'] = baselines
            save(records, "records")

        if 'nvisits' not in record:
            record['nvisits'] = 0

        record['nvisits'] += 1

        if 'exp' not in record:
            record['exp'] = []

        model.set_graph(record["graph"])

        b = np.random.choice(record['baselines'])
        features = prepare_features(record, b.nodemask, np.array(b.ncclmask), b.psmask, b.feedback)

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            nodelogit, nccllogit = model(features, training=True)

            rs = pool.map(sample_and_evaluate_with_feedback, [(record, nodelogit, nccllogit) for _ in range(64)])
            loss = 0

            # evaluate_with_feedback(record, sample_logits(nodelogit), sample_logits(nccllogit), None, trace=True)

            # score loss
            base = min(b.score for b in record['baselines'])
            mean = sum(score for _, _, score, _ in rs) / len(rs)
            for nodemask, ncclmask, loss_env, feedback in rs:
                normalized_loss = (loss_env - mean) / base
                loss += normalized_loss * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(nodemask.astype(np.float32), nodelogit))
                loss += normalized_loss * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ncclmask.astype(np.float32), nccllogit))
            loss /= len(rs)

            # similarity loss: we want the new strategy somewhat similar to the old one
            # loss += similarity_factor / np.sqrt(record['nvisits']) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(b.nodemask.astype(np.float32), nodelogit))
            # loss += similarity_factor / np.sqrt(record['nvisits']) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(np.array(b.ncclmask).astype(np.float32), nccllogit))

            # regularization loss
            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)

            info([("*" if b.invalidity > 0 else "") + str(b.score) for b in record['baselines']], min(score for _, _, score, _ in rs))

            grads = tape.gradient(loss, model.trainable_weights)
            # info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # checkpoint
        if epoch % 20 == 0:
            info("==== save ====")
            model.save_weights('weights')
            save(records, "records")
