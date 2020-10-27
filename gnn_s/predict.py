import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample, evaluate, sample_and_evaluate, reference
from search import search
from utils import save, load, info
from tge import TGE

records = load("records")

with tf.device("/gpu:1"):
    model = Model(records[0]["op_table"])
    model.load_weights('weights')

    # for record in records:
    #     bestloss, bestnode, bestnccl = record['elites'][0]
    #     for loss, node, nccl in record['elites']:
    #         if loss < bestloss:
    #             bestloss = loss
    #             bestnode = node
    #             bestnccl = nccl

    #     info(bestloss, record['reference'])

    # raise SystemExit()

    record = records[-6]
    bestloss, bestnode, bestnccl = record['elites'][0]
    for loss, node, nccl in record['elites']:
        if loss < bestloss:
            bestloss = loss
            bestnode = node
            bestnccl = nccl

    nodemask = np.reshape(bestnode, (len(record['cgroups']), len(record['devices'])))
    ncclmask = bestnccl

    info(nodemask, ncclmask)

    gdef = record["gdef"]
    strategy = { gdef.node[i].name: [int(ncclmask[gi])] + [ int(nodemask[gi, j]) for j in range(nodemask.shape[1]) ] for gi, group in enumerate(record["cgroups"]) for i in group }
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            v[1] = 1
    d = {}
    for n, s in strategy.items():
        if tuple(s) not in d:
            d[tuple(s)] = 1
        else:
            d[tuple(s)] += 1
    for s, c in d.items():
        info(s, c)

    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(120)
    tge.replace_placeholder(120)
    tge.set_bandwidth(intra=int(record["intra"]), inter=int(record["inter"]))
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"], "trace_best.json")

    strategy = { gdef.node[i].name: [1] + [ 1 for j in range(nodemask.shape[1]) ] for gi, group in enumerate(record["cgroups"]) for i in group }
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(120)
    tge.replace_placeholder(120)
    tge.set_bandwidth(intra=int(record["intra"]), inter=int(record["inter"]))
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"], "trace_dp.json")


