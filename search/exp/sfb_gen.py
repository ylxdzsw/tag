import numpy as np
import sys
import json
import copy
from utils import info, load, save
import subprocess
from data import TopoSpec, TopoSpecTask, ProfileData, gen_data, gen_random_topology,estimate_model_size
import tge
from mcts import Tree
from sfb import solve_sfb
import sys

m = sys.argv[1]

gdef = load('raw_data/{}/model.pickle'.format(m))
prof_data = ProfileData(m)
batchsize = 6
tge.simplify_graph(gdef, sinks=["Adam", "init"])

# topo = TopoSpec([
#     TopoSpecTask('1080ti', 10<<30, 3000, 1),
#     TopoSpecTask('p100', 9<<30, 3000, 2),
# ], [[2810 for _ in range(2)] for _ in range(2)])

topo = TopoSpec([
    TopoSpecTask('v100', 30<<30, 8000, 1),
    TopoSpecTask('v100', 30<<30, 8000, 1),
], [[2810 for _ in range(2)] for _ in range(2)])

record = gen_data(gdef, prof_data, batchsize, topo)

trace = []
def trace_fun(leaf_state):
    trace.append(leaf_state)
    info(leaf_state.result[0], leaf_state.actions)
Tree(record, None, real_topo=True).playout(800, trace_fun)

best_state = max(trace, key=lambda x: x.result[0])
info("best: ", best_state.result[0], best_state.actions)
strategy = best_state.dump_strategy()

x = json.loads(solve_sfb(record, strategy))
info(x['result'])
save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), f"sfb_{m}_our_without")

for i in x['result']:
    strategy[gdef.node[i].name][0] = 4

save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), f"sfb_{m}_our_with")

base_state = copy.copy(trace[0])
base_state.actions = [base_state.baseline[1]]
strategy = base_state.dump_strategy()

x = json.loads(solve_sfb(record, strategy))
info(x['result'])
save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), f"sfb_{m}_dp_without")

for i in x['result']:
    strategy[gdef.node[i].name][0] = 4

save((gdef, prof_data.to_tge(topo, batchsize), batchsize, strategy), f"sfb_{m}_dp_with")
