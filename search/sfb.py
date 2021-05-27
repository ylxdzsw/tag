import numpy as np
import sys
import json
import copy
from utils import info, load, save
import subprocess
from data import TopoSpec, TopoSpecTask, ProfileData, gen_data, gen_random_topology,estimate_model_size
import tge
from mcts import Tree

def solve_sfb(record, strategy):
    gdef, batchsize, computation_times = record['gdef'], record['batchsize'], record['computation_times']

    M = [] # replication numbers for the i-th node
    B = [] # bottleneck bandwidth in the devices of the i-th node
    T = [] # average computation time on the devices
    r = [] # r[i] == 1 means the i-th node is not considered for duplication
    for node_id, node in enumerate(gdef.node):
        s = strategy[node.name]
        m = sum(s[1:])
        M.append(m)
        B.append(2810) # TODO: real value
        T.append(np.average(computation_times[node_id, s[1:], 0]))
        if m >= 2 and s[0] == 1:
            r.append(0)
        else:
            r.append(1)

    S = {}
    grads = []

    name_dict = { node.name: i for i, node in enumerate(gdef.node) }

    for node_id, node in enumerate(gdef.node):
        for input_str in node.input:
            if input_str[0] == '^':
                continue
            input_name, input_index = parse_input(input_str)
            input_id = name_dict[input_name]
            size = get_output_size(gdef.node[input_id], input_index, batchsize)
            if (input_id, node_id) in S:
                S['{},{}'.format(input_id, node_id)] += size
            else:
                S['{},{}'.format(input_id, node_id)] = size
        if node.op == 'ApplyAdam':
            input_name, input_index = parse_input(node.input[9])
            input_id = name_dict[input_name]
            grads.append((input_id, node_id))

    return call_solver(json.dumps({'S': S, 'B': B, 'M': M, 'batchsize': batchsize, 'T': T, 'grads': grads, 'r': r}))

def call_solver(x):
    p = subprocess.Popen(['julia', 'sfb.jl'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p.communicate(input=x.encode('utf-8'))[0]

def parse_input(input_str):
    if ':' in input_str:
        input_name, input_index = input_str.split(':')
        input_index = int(input_index)
    else:
        input_name = input_str
        input_index = 0
    return input_name, input_index

def get_output_size(nodedef, index, batchsize):
    shape = [ dim.size for dim in nodedef.attr["_output_shapes"].list.shape[index].dim ]
    if len(shape) > 0 and shape[0] == -1:
        shape[0] = batchsize
    tensorsize = 1
    for size in shape:
        if size == -1:
            return 1 << 30 # a large invalid number
        tensorsize *= size
    return tensorsize

if __name__ == '__main__':
    import sys

    m = sys.argv[1]

    gdef = load('raw_data/{}/model.pickle'.format(m))
    prof_data = ProfileData(m)
    batchsize = 6
    tge.simplify_graph(gdef, sinks=["Adam", "init"])

    topo = TopoSpec([
        TopoSpecTask('1080ti', 10<<30, 3000, 1),
        TopoSpecTask('p100', 9<<30, 3000, 2),
    ], [[2810 for _ in range(2)] for _ in range(2)])

    record = gen_data(gdef, prof_data, batchsize, topo)

    trace = []
    def trace_fun(leaf_state):
        trace.append(leaf_state)
        info(leaf_state.result[0], leaf_state.actions)
    Tree(record, None, real_topo=True).playout(50, trace_fun)

    best_state = max(trace, key=lambda x: x.result[0])
    info("best: ", best_state.result[0], best_state.actions)
    strategy = best_state.dump_strategy()

    x = json.loads(solve_sfb(record, strategy))
    info(x['result'])

    base_state = copy.copy(trace[0])
    base_state.actions = [base_state.baseline[1]]
    strategy = base_state.dump_strategy()

    x = json.loads(solve_sfb(record, strategy))
    info(x['result'])
