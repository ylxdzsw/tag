import numpy as np
import itertools
import copy
import tge
from data import TopoSpec, TopoSpecTask, ProfileData, gen_data, gen_random_topology,estimate_model_size
from grouping import group_with_topk_nodes, group_with_tge_basegroups
from utils import info, load
from metis import metis
from environment import evaluate_with_feedback, invalidity
from mcts import State, Tree

if __name__ == '__main__':
    import sys

    m = sys.argv[1]

    gdef = load('raw_data/{}/model.pickle'.format(m))
    prof_data = ProfileData(m)
    tge.simplify_graph(gdef, sinks=["Adam"])

    model_size = estimate_model_size(gdef, prof_data.maximum_batchsize())
    topo = gen_random_topology(model_size)
    info(topo)

    record = gen_data(gdef, prof_data, prof_data.maximum_batchsize(), topo)

    state = State.new(record)
    trace = []
    def trace_fun(value, actions):
        trace.append((value, actions))
        info(value, actions)
    info(Tree(None).playout(state, 5000, trace_fun))

# tsp sh -c 'python generalization.py vgg > vgg0.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg1.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg2.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg3.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg4.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg5.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg6.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg7.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg8.mcts 2>/dev/null'
# tsp sh -c 'python generalization.py vgg > vgg9.mcts 2>/dev/null'
