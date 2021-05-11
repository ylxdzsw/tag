from dataclasses import dataclass

from utils import groupby, car, cadr, cdr, info, load, parse_input, get_input_size

import math
import ctypes

libmetis = ctypes.cdll.LoadLibrary("./libmetis.so")

libmetis.METIS_PartGraphKway.argtypes = [
    ctypes.POINTER(ctypes.c_int64), # nvtxs
    ctypes.POINTER(ctypes.c_int64), # ncon
    ctypes.POINTER(ctypes.c_int64), # xadj
    ctypes.POINTER(ctypes.c_int64), # adjncy
    ctypes.POINTER(ctypes.c_int64), # vwgt
    ctypes.POINTER(ctypes.c_int64), # vsize
    ctypes.POINTER(ctypes.c_int64), # adjwgt
    ctypes.POINTER(ctypes.c_int64), # nparts
    ctypes.POINTER(ctypes.c_float), # tpwgts
    ctypes.POINTER(ctypes.c_float), # ubvec
    ctypes.POINTER(ctypes.c_int64), # options
    ctypes.POINTER(ctypes.c_int64), # objval
    ctypes.POINTER(ctypes.c_int64), # part
]

libmetis.METIS_PartGraphKway.restype = ctypes.c_int

@dataclass
class MetisNode:
    raw_id: int
    adj: list # (new_id, tensorsize)
    computation: int
    memory: int

def metis(gdef, prof_data, npart, nodes, batchsize, computation_balance_factor=5.0, memory_balance_factor=2.0):
    nodes = [ MetisNode(i, [], 0, 1) for i in nodes ]
    raw_to_new_map = { node.raw_id: i for i, node in enumerate(nodes) }
    name_dict = { node.name: i for i, node in enumerate(gdef.node) }

    for new_id, node in enumerate(nodes):
        node_def = gdef.node[node.raw_id]
        for input_index, input_tensor_name in enumerate(node_def.input):
            input_name, input_tensor_index = parse_input(input_tensor_name)
            input_raw_id = name_dict[input_name]

            if input_raw_id not in raw_to_new_map:
                continue

            input_new_id = raw_to_new_map[input_raw_id]
            input_node = nodes[input_new_id]
            input_node_def = gdef.node[input_raw_id]
            tensorsize = get_input_size(input_node_def, input_tensor_index, batchsize)
            if tensorsize == 0: # TODO: differentiate the cases of invalid cut and free cut (control dependency or empty tensor)
                tensorsize = 1000

            node.adj.append((input_new_id, tensorsize))
            input_node.adj.append((new_id, tensorsize))

            if node_def.op.startswith('Apply') and input_index == 0: # this input is a variable tensor
                input_node.memory = 1 + int(math.sqrt(tensorsize))

        # node.computation = prof_data[(node_def.name,1)][0]

    xadj = []
    adjncy = []
    vwgt = []
    adjwgt = []
    ubvec = [memory_balance_factor]

    for node in nodes:
        # info(node)
        xadj.append(len(adjncy))
        for adj_id, tensorsize in node.adj:
            adjncy.append(adj_id)
            adjwgt.append(tensorsize)
        vwgt.append(node.memory)
    xadj.append(len(adjncy))

    objvar = (ctypes.c_int64 * 1)(0)
    membership = (ctypes.c_int64*len(nodes))(*[0]*len(nodes))

    ret = libmetis.METIS_PartGraphKway(
        ctypes.byref(ctypes.c_int64(len(nodes))), # nvtxs
        ctypes.byref(ctypes.c_int64(1)), # ncon
        ctypes.cast((ctypes.c_int64*len(xadj))(*xadj), ctypes.POINTER(ctypes.c_int64)), # xadj
        ctypes.cast((ctypes.c_int64*len(adjncy))(*adjncy), ctypes.POINTER(ctypes.c_int64)), # adjncy
        ctypes.cast((ctypes.c_int64*len(vwgt))(*vwgt), ctypes.POINTER(ctypes.c_int64)), # vwgt
        None, # vsize
        ctypes.cast((ctypes.c_int64*len(adjwgt))(*adjwgt), ctypes.POINTER(ctypes.c_int64)), # adjwgt
        ctypes.byref(ctypes.c_int64(npart)), # nparts
        None, # tpwgts
        ctypes.cast((ctypes.c_float*len(ubvec))(*ubvec), ctypes.POINTER(ctypes.c_float)), # ubvec
        None, # options
        ctypes.cast(objvar, ctypes.POINTER(ctypes.c_int64)), # objval
        ctypes.cast(membership, ctypes.POINTER(ctypes.c_int64)), # part
    )

    assert ret == 1

    return list(objvar)[0], list(membership)
