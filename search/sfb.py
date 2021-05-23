import sys
import json
from utils import load

name = sys.argv[1]

S = {}
grads = []
B = 2810
M = 2

gdef, batchsize, prof_dict = load("{}.pickle".format(name))

name_dict = { node.name: i for i, node in enumerate(gdef.node) }
T = [ prof_dict[node.name] for node in gdef.node ]

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

with open("{}.json".format(name), 'w') as f:
    json.dump({'S': S, 'B': B, 'M': M, 'batchsize': batchsize, 'T': T, 'grads': grads}, f)

