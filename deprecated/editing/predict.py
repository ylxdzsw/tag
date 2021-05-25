import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import GNN, run_episode, get_expected_return
from environment import replication_number_feasibility_rounding, evaluate_with_feedback
from utils import save, load, info
from tge import TGE


records = load("records")
record_id = 1 # np.random.randint(len(records))
record = records[record_id]


with tf.device("/gpu:1"):
    model = GNN()
    model.load_weights('weights')
    model.set_graph(record["graph"], record["segments"])

    baseline = record['baselines'][np.random.randint(len(record['baselines']))]
    initial_state = baseline.nodemask, baseline.ncclmask, baseline.psmask, baseline.score, baseline.feedback

    action_probs, action_entropys, values, rewards = run_episode(record, initial_state, model)
    info([("*" if b.invalidity > 0 else "") + str(b.score) for b in record['baselines']], rewards[-1])


    # for record in records:
    #     info([("*" if b.invalidity > 0 else "") + str(b.score) for b in record['baselines']], record['traces'][-1].score)
        # scores = sorted([b.score for b in record['baselines']])

    raise SystemExit()

    record = records[0]

    gdef = record["gdef"]

    trace = record['traces'][-1]
    # replication_number_feasibility_rounding(record, nodemask)
    evaluate_with_feedback(record, trace.nodemask, trace.ncclmask, [0]*trace.nodemask.shape[0], True)

    # dp = record['baselines'][1]
    # evaluate_with_feedback(record, dp.nodemask, dp.ncclmask, [0]*dp.nodemask.shape[0], "trace_dp.json")



