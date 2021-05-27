from dataclasses import dataclass
from typing import Any
import numpy as np
import time
import tensorflow as tf
import collections

from model import Model, policy
from environment import evaluate_with_feedback, invalidity
from utils import save, load, info
from mcts import Tree, Action

import os
import sys

out_folder = sys.argv[1]
os.mkdir(out_folder)

data = []

for f in os.listdir("init_data")[:20]:
    data.extend(load(f"init_data/{f}"))

with tf.device("/gpu:3"):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=.6) # https://openreview.net/pdf?id=r1etN1rtPB

    model = Model()

    acc = 0
    for epoch in range(len(data) * 2000):
        state, actions, probs = data[np.random.randint(len(data))]

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)

            log_p = policy(model, state, actions, use_runtime_feedback=True)
            loss = -tf.math.reduce_sum(probs * log_p)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        acc += loss.numpy()
        if epoch % 100 == 99:
            info(acc)
            model.save_weights(f"{out_folder}/loss_curve_weights_{epoch}_{acc}")
            acc = 0

