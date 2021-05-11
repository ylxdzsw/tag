from dataclasses import dataclass
from typing import Any
import numpy as np
import time
import tensorflow as tf
import collections

from model import GNN, Decoder
from environment import evaluate_with_feedback, score, invalidity
from utils import save, load, info
from baseline import gen_baselines, eval_baselines, random_cross_node, random_shuffle_node, Trace
from mcts import State, Action

from multiprocessing import Pool

# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

def playout(record, gnn, decoder):
    gdef, topo_spec, prof_data = record['gdef'], record['topo_spec'], record['prof_data']
    batchsize = prof_data.maximum_batchsize()

    sorted_groups = sorted(record['op_groups'], key=lambda group: -np.sum([ prof_data.get('1080ti', batchsize)[gdef.node[node_id].name] for node_id in group ])) # largest computation time first
    state = State(record, sorted_groups, 0, [])

    state_copy = state.clone()
    state_copy.actions.append(([1 for _ in range(len(topo_spec.tasks))], 1))
    time, _ = evaluate_with_feedback(state_copy)

    # TODO: if OOM, use MP as baseline
    # TODO: save to record
    state.dp_time = time

    mcts = Tree(None).playout(state, 50000)

    data = [] # (actions, action_probs)

    while True:
        status = game.get_status()
        if status != 0:
            if status == 1:
                return [ (*x, 1 if i%2 == 0 else -1) for i, x in enumerate(data) ]
            if status == 2:
                return [ (*x, -1 if i%2 == 0 else 1) for i, x in enumerate(data) ]
            if status == 3:
                return [ (*x, 0) for x in data ]
            return record, status

        mcts.playout(game, 800 - mcts.total_visits())

        action_probs = mcts.get_action_probs(0.1)
        pieces, mask, probs = encode_input(game, action_probs)

        data.append((pieces, mask, probs))

        # state = game.dump()
        # value = mcts.root_value()
        # print(state, action_probs, value)

        old_pos, new_pos = mcts.sample_action(0.2, 0.1)
        game.do_move(old_pos, new_pos)
        mcts.chroot(old_pos, new_pos)

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
    with Pool(8, initializer=worker_init, initargs=('scripted_model.pt',)) as pool:
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
