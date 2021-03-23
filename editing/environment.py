from tge import TGE
from utils import car, cadr, cdr, info
from metis import metis
import numpy as np
import tensorflow as tf
import tempfile
import os
import json

def sample_logits(logit, e=0):
    p = tf.math.sigmoid(logit)
    def f(x):
        if np.random.rand() < e:
            return np.random.choice(2)
        else:
            return int(np.random.rand() < x)
    return np.vectorize(f)(p)

def sample_and_evaluate_with_feedback(pack):
    record, nodelogit, nccllogit = pack
    nodemask = sample_logits(nodelogit)
    ncclmask = sample_logits(nccllogit)
    return (nodemask, ncclmask, *evaluate_with_feedback(record, nodemask, ncclmask, None))

def evaluate_with_feedback(record, nodemask, ncclmask, psmask, trace=""):
    gdef = record["gdef"]
    # replication_number_feasibility_rounding(record, nodemask)
    strategy = {}
    for node_id in range(len(gdef.node)):
        s = [0 if int(ncclmask[node_id]) == 0 else int(ncclmask[node_id])]
        for i in range(nodemask.shape[1]):
            s.append(int(nodemask[node_id, i]))
        strategy[gdef.node[node_id].name] = s
    # info(strategy)
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            v[1] = 1
    tge = TGE(gdef, [device for device, _ in record["devices"]], sinks=["Adam"])
    tge.set_strategy(strategy)
    tge.fill_batchsize(record["batchsize"])
    tge.replace_placeholder(record["batchsize"])
    tge.set_topology(*record["topology_for_simulator"])
    tge.set_nccl_model(record["nccl_models"])

    temp_path = tempfile.mktemp()
    time, mem = tge.evaluate(record["prof_data"].to_tge(record["topo_spec"], record["batchsize"]), chrome_path=trace, dump_path=temp_path)
    feedback = parse_feedback(temp_path)
    feedback["peak_memory"] = mem
    feedback["leftout"] = list((np.sum(nodemask, axis=1) == 0).astype(int))
    os.remove(temp_path)

    return score(record, time, feedback), feedback

def parse_feedback(temp_path):
    raw = json.load(open(temp_path, "r"))
    return raw

def invalidity(record, feedback): # 0 means valid
    oom = 0
    for peak_memory, (_, task_id) in zip(feedback["peak_memory"], record["devices"]):
        if peak_memory > record["topo_spec"].tasks[task_id].memory:
            oom += 1
    return oom + sum(feedback["leftout"])

def score(record, time, feedback):
    return np.sqrt(time / 1_000_000) * (1 + 10 * invalidity(record, feedback))

def replication_number_feasibility_rounding(record, nodemask):
    B = record["batchsize"]

    for i in range(nodemask.shape[0]):
        r = sum(nodemask[i, :])
        if B % r == 0:
            continue

        actions = []
        if r > 1 and B % (r - 1) == 0: # try remove one replica. Devices that have two replicas has double probability
            for j, k in enumerate(nodemask[i, :]):
                actions.extend([(j, -1)] * k)
        if B % (r + 1) == 0: # try add one replica, but only on devices that already have one
            for j, k in enumerate(nodemask[i, :]):
                if k == 1:
                    actions.append((j, 1))

        if len(actions) < 0: # heuristic failed, randomly remove replicas until feasible
            while B % sum(nodemask[i, :]) != 0:
                j = np.random.chioce(nodemask.shape[1])
                if nodemask[i, j] > 0:
                    nodemask[i, j] -= 1
            continue

        j, a = np.random.choice(actions)
        nodemask[i, j] += a

    return nodemask
