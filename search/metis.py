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
    gid: int
    adj: dict # new_id => tensorsize
    vwgt: int

def metis(gdef, base_groups, node_cost_list, npart, target_nodes, batchsize, balance_factor=2.):
    metis_nodes = [ MetisNode(gid, [], 1) for gid, group in enumerate(base_groups) if any((i in target_nodes) for i in group) ]

    raw_to_group_map = { i: gid for gid, group in enumerate(base_groups) for i in group }
    group_to_metis_map = { metis_node.gid: metis_id for metis_id, metis_node in enumerate(metis_nodes) }
    name_to_raw_map = { node_def.name: raw_id for raw_id, node_def in enumerate(gdef.node) }

    if len(metis_nodes) <= npart: # not enough nodes to partition, metis will complaint.
        return [ group_to_metis_map[raw_to_group_map[raw_id]] for raw_id in target_nodes ]

    for target_index, raw_id in enumerate(target_nodes):
        metis_id = group_to_metis_map[raw_to_group_map[raw_id]]
        metis_node = metis_nodes[metis_id]
        node_def = gdef.node[raw_id]

        for input_index, input_tensor_name in enumerate(node_def.input):
            input_name, input_tensor_index = parse_input(input_tensor_name)
            input_group_id = raw_to_group_map[name_to_raw_map[input_name]]

            if input_group_id == metis_node.gid:
                continue

            if input_group_id not in group_to_metis_map:
                continue

            input_metis_id = group_to_metis_map[input_group_id]
            input_metis_node = metis_nodes[input_metis_id]
            input_node_def = gdef.node[input_raw_id]

            tensorsize = get_input_size(input_node_def, input_tensor_index, batchsize)
            if tensorsize == 0: # TODO: differentiate the cases of invalid cut and free cut (control dependency or empty tensor)
                tensorsize = 1000

            if input_metis_id not in node.adj:
                metis_node.adj[input_metis_id] = 0
                input_metis_node.adj[metis_id] = 0

            metis_node.adj[input_metis_id] += tensorsize
            input_metis_node.adj[metis_id] += tensorsize

        metis_node.vwgt += node_cost_list[target_index]

    xadj = []
    adjncy = []
    vwgt = []
    adjwgt = []
    ubvec = [balance_factor]

    for metis_node in metis_nodes:
        # info(metis_node)
        xadj.append(len(adjncy))
        for adj_id, tensorsize in metis_node.adj:
            adjncy.append(adj_id)
            adjwgt.append(tensorsize)
        vwgt.append(metis_node.vwgt)
    xadj.append(len(adjncy))

    objvar = (ctypes.c_int64 * 1)(0)
    membership = (ctypes.c_int64*len(metis_nodes))(*[0]*len(metis_nodes))

    ret = libmetis.METIS_PartGraphKway(
        ctypes.byref(ctypes.c_int64(len(metis_nodes))), # nvtxs
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

    return [ membership[group_to_metis_map[raw_to_group_map[raw_id]]] for raw_id in target_nodes ]
