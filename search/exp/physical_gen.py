import numpy as np
import itertools
import copy
import tge
from data import TopoSpec, TopoSpecTask, ProfileData, gen_data, gen_random_topology,estimate_model_size
from grouping import group_with_topk_nodes, group_with_tge_basegroups
from utils import info, load, save
from metis import metis
from environment import evaluate_with_feedback, invalidity
from mcts import Tree
from model import Model, policy

if __name__ == '__main__':
    import sys

    m = sys.argv[1]

    gdef = load('raw_data/{}/model.pickle'.format(m))
    prof_data = ProfileData(m)
    batchsize = min(max(prof_data.maximum_batchsize() * 2, 32), 600)
    tge.simplify_graph(gdef, sinks=["Adam", "init"])

    topo = TopoSpec([
        TopoSpecTask('v100', 30<<30, 8000, 4),
        TopoSpecTask('1080ti', 9<<30, 3000, 8),
        TopoSpecTask('p100', 10<<30, 3000, 4),
        # TopoSpecTask('t4', 14<<30, 3000, 16),
    ], [[2810 for _ in range(3)] for _ in range(3)])

    record = gen_data(gdef, prof_data, batchsize, topo)

    model = Model()
    model.load_weights(sys.argv[2])

    trace = []
    def trace_fun(leaf_state):
        trace.append(leaf_state)
        info(leaf_state.result[0], leaf_state.actions)
    # Tree(record, None, real_topo=True).playout(2000, trace_fun)
    Tree(record, lambda state, actions: policy(model, state, actions), real_topo=True).playout(2000, trace_fun)

    for ntimes in (50, 800, 2000):
        best_state = max(trace[:ntimes], key=lambda x: x.result[0])
        info("best: ", ntimes, best_state.result[0], best_state.actions)
        strategy = best_state.dump_strategy()
        save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), "{}_strategy_{}".format(m, ntimes))

    base_state = copy.copy(trace[0])
    base_state.actions = [base_state.baseline[1]]
    strategy = base_state.dump_strategy()
    save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), "{}_baseline".format(m))

