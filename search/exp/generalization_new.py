import numpy as np
import itertools
import copy
import tge
from data import TopoSpec, TopoSpecTask, ProfileData, gen_data
from grouping import group_with_topk_nodes, group_with_tge_basegroups
from utils import info, load, save
from metis import metis
from environment import evaluate_with_feedback, invalidity
from mcts import State, Tree

def gen_random_topology(target_n_devices=8):
    gpu_models = ('p100', '1080ti', 'v100')
    gpu_memory = {
        'p100': 10<<30,
        '1080ti': 9<<30,
        'v100': 14<<30
    }
    intra_links = (8000, 20000) # PCI, nvlink
    inter_links = (2810, 5000)
    card_numbers = (1, 2, 3, 4)

    tasks = []
    intra_bandwidth = np.random.choice(intra_links)
    while True:
        if sum(task.number for task in tasks) >= target_n_devices:
            return None

        gpu_model = np.random.choice(gpu_models)
        memory = gpu_memory[gpu_model]
        card_number = np.random.choice(card_numbers)
        task = TopoSpecTask(gpu_model, memory, intra_bandwidth, card_number)

        tasks.append(task)

        if sum(task.number for task in tasks) == target_n_devices:
            break

    n = len(tasks)
    inter_bandwidth = [ [0 for _ in range(n) ] for _ in range(n) ]
    inter_link = np.random.choice(inter_links)
    for i in range(n):
        for j in range(i+1):
            inter_bandwidth[i][j] = inter_link
            inter_bandwidth[j][i] = inter_link

    return TopoSpec(tasks, inter_bandwidth)

if __name__ == '__main__':
    import sys

    i = int(sys.argv[1])

    m = np.random.choice(['vgg', 'berts', 'transformer', 'resnet', 'inception'])

    gdef = load('raw_data/{}/model.pickle'.format(m))
    prof_data = ProfileData(m)
    tge.simplify_graph(gdef, sinks=["Adam"])

    topo = None
    while topo is None:
        topo = gen_random_topology()
    assert sum(task.number for task in topo.tasks) == 8

    record = gen_data(gdef, prof_data, prof_data.maximum_batchsize(), topo)

    state = State.new(record)

    save((
        gdef, # graphdef
        [device for device, _ in record['device_list']], # device list
        topo.tasks[0].intra_bandwidth, # intra bandwidth
        topo.bandwidth[0][0], # inter bandwidth
        record["nccl_models"], # nccl model
        record["batchsize"], # batch size
        prof_data.to_tge(topo, record["batchsize"]), # prof data
        state.baseline[0]
    ), f"generalization_exp/data_{i}")

    trace = []
    def trace_fun(state):
        trace.append(state.result[0])
    Tree(record, None).playout(2000, trace_fun)

    bf = [ i for i, v in enumerate(trace) if v > 0 ]
    if len(bf) > 0:
        bf = bf[0]
    else:
        bf = 2000
    a = max(trace[:50])
    b = max(trace)

    info(bf, a, b)



