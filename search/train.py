from dataclasses import dataclass
from typing import Any
import numpy as np
import time
import tensorflow as tf
import collections

from model import Model, policy
from environment import evaluate_with_feedback, score, invalidity
from utils import save, load, info
from baseline import gen_baselines, eval_baselines, random_cross_node, random_shuffle_node, Trace
from mcts import State, Action

from multiprocessing import Pool

# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

def playout(record, model):
    state = State.new(record)
    mcts = Tree(lambda state, actions: policy(model, state, actions)).playout(state, 5000)

    data = [] # (state, actions, probs)

    for i in range(len(record['op_groups']) // 2):
        mcts.playout(game, 800 - mcts.root.n_visits)
        actions, probs = mcts.get_actions_and_probs()
        data.append((state.clone(), actions, probs))
        mcts.chroot(np.random.choice(range(len(probs)), probs))

    return data

def worker_init(model_path):
    global model
    torch.set_num_threads(1)
    import os
    set_random_seed(os.getpid() * 7 + 39393)
    model = torch.jit.load(model_path).eval()

def worker_run(board_type):
    global model
    return self_play(board_type, model)

def collect_data(record):
    model.cpu().save('scripted_model.pt')
    with Pool(40, initializer=worker_init, initargs=('scripted_model.pt',)) as pool:
        data_batches = pool.map(worker_run, (board_type for _ in range(n)), chunksize=1)
    return [ x for batch in data_batches for x in batch ]

def train(model, optimizer, data):
    model.train()

    acc = 0, 0
    for epoch in range(len(data) // 4):
        # pieces, masks, probs, scores = ( torch.from_numpy(x).cuda() for x in random_batch(data, 32) )
        pieces, masks, probs, scores = ( torch.from_numpy(x) for x in random_batch(data, 32) )
        policy, value = model(pieces, masks)
        policy_loss = -torch.mean(torch.sum(probs * policy, 1))
        value_loss = torch.nn.functional.mse_loss(value, scores.float())

        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .6)
        optimizer.step()
        epoch += 1

        acc = acc[0] + policy_loss.item() / 100, acc[1] + value_loss.item() / 100
        if epoch % 100 == 99:
            print(*acc)
            acc = 0, 0

records = load("records")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=.6) # https://openreview.net/pdf?id=r1etN1rtPB

with tf.device("/gpu:0"):
    gnn = GNN()
    decoder = Decoder()

    try:
        gnn.load_weights('gnn_weights')
        decoder.load_weights('decoder_weights')
        info("load saved weight")
    except:
        info("no saved weight")



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
