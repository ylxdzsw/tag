from dataclasses import dataclass
from typing import Any
import numpy as np
import time
import tensorflow as tf
import collections
import sys

from model import Model, policy
from environment import evaluate_with_feedback, invalidity
from utils import save, load, info
from mcts import Tree, Action

from multiprocessing import Pool

# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

def playout(record, model=None):
    if model is not None:
        model.set_graph(record["graph"]) # the graph won't change during playout
        mcts = Tree(record, lambda state, actions: policy(model, state, actions))
    else:
        mcts = Tree(record, None)

    data = [] # (state, actions, probs)
    for i in range(len(record['op_groups']) // 2):
        mcts.playout(8 - mcts.root.n_visits)
        actions, probs = mcts.get_actions_and_probs()
        data.append((mcts.root.state, actions, probs)) # no need to clone: it is about to be freed
        mcts.chroot(np.random.choice(range(len(probs)), p=probs))

    return data

def worker_init(model_path):
    tf.config.experimental.set_visible_devices([], 'GPU')

    global model
    model = Model()
    model.load_weights(model_path)

def worker_run(record):
    global model
    return playout(record, model)

def collect_data(records, model):
    model.save('_weights')
    with Pool(40, initializer=worker_init, initargs=('_weights',)) as pool:
        data_batches = pool.map(worker_run, (records[i % len(records)] for i in range(100)), chunksize=1)
    return [ x for batch in data_batches for x in batch ]

L2_regularization_factor = 0

def train(model, optimizer, data):
    acc = 0
    for epoch in range(len(data) * 4):
        state, actions, probs = data[np.random.randint(len(data))]

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)

            log_p = policy(model, state, actions)
            loss = -tf.math.reduce_mean(probs * log_p)

            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        acc += loss.numpy()
        if epoch % 100 == 99:
            print(acc)
            acc = 0

records = load("records")

with tf.device("/gpu:0"):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=.6) # https://openreview.net/pdf?id=r1etN1rtPB

    model = Model()

    if len(sys.argv) > 1:
        r = int(sys.argv[1])
        model.load_weights(f"weights_{r}")
        info(f"continue from {r} round")
    else:
        init_data = playout(records[0])
        train(model, optimizer, init_data)
        r = 0

    while True:
        data = collect_data(records, model)
        train(model, optimizer, data)

        r += 1

        info("==== save ====")
        model.save_weights(f"weights_{r}")
        save(records, "records")
