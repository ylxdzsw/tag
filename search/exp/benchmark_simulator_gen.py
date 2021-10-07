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

if __name__ == '__main__':
    import sys

    m = sys.argv[1]

    gdef = load('raw_data/{}/model.pickle'.format(m))
    prof_data = ProfileData(m)
    batchsize = min(max(prof_data.maximum_batchsize() * 2, 32), 600)
    tge.simplify_graph(gdef, sinks=["Adam", "init"])

    topo = TopoSpec([
        TopoSpecTask('1080ti', 12<<30, 3000, 2),
        TopoSpecTask('p100', 10<<30, 3000, 2),
    ], [[2810 for _ in range(4)] for _ in range(4)])

    options = [[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]]
    strategy = { node.name: [np.random.randint(0, 2)] + options[np.random.randint(0, len(options))] for node in gdef.node }

    save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), "{}_test".format(m))

