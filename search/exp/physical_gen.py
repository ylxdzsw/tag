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
    batchsize = prof_data.maximum_batchsize()
    tge.simplify_graph(gdef, sinks=["Adam", "init"])

    topo = TopoSpec([
        TopoSpecTask('1080ti', 9<<30, 8000, 2),
        TopoSpecTask('1080ti', 9<<30, 8000, 2),
        TopoSpecTask('v100',   12<<30, 8000, 4),
    ], [[4000, 4000, 4000],
        [4000, 4000, 4000],
        [4000, 4000, 4000]])

    record = gen_data(gdef, prof_data, batchsize, topo)

    state = State.new(record)
    trace = []
    def trace_fun(value, actions):
        trace.append((value, actions))
        info(value, actions)
    best, best_actions = Tree(None, real_topo=True).playout(state, 500, trace_fun)
    info("best:", best, best_actions)

    state.actions = best_actions
    strategy = state.dump_strategy()

    save((gdef, prof_data.to_tge(topo, record['batchsize']), batchsize, strategy), "{}_strategy".format(m))

    state.actions = [([1 for _ in range(len(topo.tasks))], 1)]
    strategy = state.dump_strategy()
    save((gdef, prof_data.to_tge(topo, record['batchsize']), batchsize, strategy), "{}_dp".format(m))

