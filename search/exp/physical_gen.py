import numpy as np
import itertools
import copy
import tge
from data import TopoSpec, TopoSpecTask, ProfileData, gen_data, gen_random_topology,estimate_model_size
from grouping import group_with_topk_nodes, group_with_tge_basegroups
from utils import info, load, save
from metis import metis
from environment import evaluate_with_feedback, invalidity
from mcts import State, Tree

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
    ], [[2810 for _ in range(3)] for _ in range(3)])

    record = gen_data(gdef, prof_data, batchsize, topo)

    state = State.new(record)
    trace = []
    def trace_fun(value, actions):
        trace.append((value, actions))
        info(value, actions)
    Tree(None, real_topo=True).playout(state, 2000, trace_fun)

    for ntimes in (50, 800, 2000):
        value, actions = max(trace[:ntimes], key=lambda x: x[0])
        info("best: ", ntimes, value, actions)
        state.actions = actions
        strategy = state.dump_strategy()
        save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), "{}_strategy_{}".format(m, ntimes))

    state.actions = [state.baseline[1]]
    strategy = state.dump_strategy()
    save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), "{}_baseline".format(m))

