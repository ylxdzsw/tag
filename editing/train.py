from dataclasses import dataclass
from typing import Any
import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample_logits, evaluate_with_feedback, score, invalidity, sample_and_evaluate_with_feedback
from utils import save, load, info
from baseline import gen_baselines, eval_baselines, random_cross_node, random_shuffle_node, Trace

from multiprocessing import Pool
pool = Pool(8)

try:
    records = load("records")
    info("load saved records")
except:
    records = get_all_data()
    info("no saved records")
    save(records, "records")

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
        np.reshape(ncclmask, (-1, 1)),
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=.00006, clipnorm=.6) # https://openreview.net/pdf?id=r1etN1rtPB
    L2_regularization_factor = 0 #.00001
    similarity_regularization_factor = 0 # .02

    for epoch in range(20000):
        record_id = np.random.randint(len(records))
        record = records[record_id]

        if 'baselines' not in record:
            baselines = gen_baselines(record)
            eval_baselines(record, baselines)
            record['baselines'] = baselines
            save(records, "records")

        if 'nvisits' not in record:
            record['nvisits'] = 0

        record['nvisits'] += 1

        if 'traces' not in record:
            record['traces'] = []
            for _ in range(10): # TODO: concurrent
                t = Trace(*random_cross_node(record, *((b.nodemask, b.ncclmask, b.psmask) for b in record['baselines'])))
                t.evaluate_with_feedback(record)
                record['traces'].append(t)

        model.set_graph(record["graph"])

        trace = record['traces'][np.random.randint(len(record['traces']))]
        features = prepare_features(record, trace.nodemask, trace.ncclmask, trace.psmask, trace.feedback)

        nodelogit, nccllogit = model(features, training=False)
        rs = pool.map(sample_and_evaluate_with_feedback, [(record, nodelogit, nccllogit) for _ in range(1024)])

        positive = 0
        for nodemask, ncclmask, score, feedback in rs:
            if score < record['traces'][-1].score:
                record['traces'].append(Trace(nodemask, ncclmask, [0] * nodemask.shape[0], score, feedback))
            if score < trace.score:
                positive += 1
        info("positive: ", positive)
        if positive == 0 and len(record['traces']) < 20:
            for nodemask, ncclmask, score, feedback in rs:
                record['traces'].append(Trace(nodemask, ncclmask, [0] * nodemask.shape[0], score, feedback))
        if positive <= 24 or positive >= 1000:
            continue

        examples = []
        for nodemask, ncclmask, score, feedback in rs:
            log_p = (
                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(np.array(nodemask).astype(np.float32), nodelogit)) +
                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(np.array(ncclmask).astype(np.float32), nccllogit))
            ).numpy()
            examples.append((log_p, nodemask, ncclmask, score))

        for _ in range(16):
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_weights)
                nodelogit, nccllogit = model(features, training=True)

                # objective
                loss = 0
                in_trusted_range = 0
                for log_p, nodemask, ncclmask, score in examples:
                    new_log_p = (
                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(np.array(nodemask).astype(np.float32), nodelogit)) +
                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(np.array(ncclmask).astype(np.float32), nccllogit))
                    )
                    L = -1 if score < trace.score else 1
                    ratio = tf.exp(new_log_p - log_p)
                    if 0.9 < ratio.numpy() < 1.1:
                        in_trusted_range += 1
                        loss += L * ratio
                if in_trusted_range > len(examples) / 2:
                    loss /= in_trusted_range
                else:
                    break

                # similarity loss: we want the new strategy somewhat similar to the old one
                if similarity_regularization_factor > 0:
                    loss += similarity_factor / np.sqrt(record['nvisits']) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(b.nodemask.astype(np.float32), nodelogit))
                    loss += similarity_factor / np.sqrt(record['nvisits']) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(np.array(b.ncclmask).astype(np.float32), nccllogit))

                # L2 regularization loss
                if L2_regularization_factor > 0:
                    for weight in model.trainable_weights:
                        loss += L2_regularization_factor * tf.nn.l2_loss(weight)

                # info([("*" if b.invalidity > 0 else "") + str(b.score) for b in record['baselines']], trace.next.score)
                # info(record_id, loss)

                grads = tape.gradient(loss, model.trainable_weights)
                # info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # checkpoint
        if epoch % 20 == 0:
            info("==== save ====")
            model.save_weights('weights')
            save(records, "records")
