from dataclasses import dataclass
from typing import Any
import numpy as np
import time
import tensorflow as tf
import collections

from data import get_all_data
from model import GNN, train_step
from environment import evaluate_with_feedback, score, invalidity
from utils import save, load, info
from baseline import gen_baselines, eval_baselines, random_cross_node, random_shuffle_node, Trace

# from multiprocessing import Pool
# pool = Pool(8)

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

try:
    records = load("records")
    info("load saved records")
except:
    records = get_all_data()
    info("no saved records")
    save(records, "records")

optimizer = tf.keras.optimizers.Adam(learning_rate=.000006, clipnorm=.6) # https://openreview.net/pdf?id=r1etN1rtPB

with tf.device("/gpu:0"):
    model = GNN()

    try:
        model.load_weights('weights')
        info("load saved weight")
    except:
        info("no saved weight")

    episode_rewards = collections.deque(maxlen=200)

    for epoch in range(20000):
        record_id = 1 # np.random.randint(len(records))
        record = records[record_id]

        if 'baselines' not in record:
            baselines = gen_baselines(record)
            eval_baselines(record, baselines)
            record['baselines'] = baselines
            save(records, "records")

        if 'nvisit' not in record:
            record['nvisit'] = 0

        if 'seeds' not in record:
            record['seeds'] = []
            for baseline in record['baselines']:
                initial_state = baseline.nodemask, baseline.ncclmask, baseline.psmask, baseline.score, baseline.feedback
                record['seeds'].append(initial_state)

        record['nvisit'] += 1

        model.set_graph(record["graph"], record["segments"])

        episode_reward = train_step(record, model, optimizer)
        episode_rewards.append(episode_reward)
        info(f'Episode {epoch}: average reward: {np.mean(episode_rewards)}')

        # checkpoint
        if epoch % 50 == 0:
            info("==== save ====")
            model.save_weights('weights')
            save(records, "records")
