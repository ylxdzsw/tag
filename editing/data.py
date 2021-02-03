from dataclasses import dataclass

import dgl
import re
import numpy as np
import pickle
import math
import itertools
import networkx as nx
from utils import groupby, car, cadr, cdr, info, load, parse_input, get_input_size
from metis import metis

@dataclass
class TopoSpec:
    tasks: list
    bandwidth: list # symetric matrix for between-task bandwidth

    @property
    def ntasks(self):
        return len(self.tasks)

@dataclass
class TopoSpecTask:
    gpu_model: str
    memory: int
    intra_bandwidth: int
    number: int

BASE_NCCL_MODEL = [0.043420241077615454, 368.2013618677043, 0.27766802543921265, 211.91926070037152] # f: size -> time

def gen_data(gdef, prof_data, batchsize, topo_spec: TopoSpec):
    devices = [(device_name(task_id, index), task_id) for task_id, task in enumerate(topo_spec.tasks) for index in range(task.number)]
    device_feats = np.array([[topo_spec.tasks[task_id].memory] for _, task_id in devices], dtype=float)

    edge_link = [], []
    link_feats = []

    for device_id, (_, task_id) in enumerate(devices):
        for another_device_id, (_, another_task_id) in enumerate(devices):
            if another_device_id == device_id:
                continue
            edge_link[0].append(device_id)
            edge_link[1].append(another_device_id)
            edge_link[0].append(another_device_id)
            edge_link[1].append(device_id)

            if another_task_id == task_id:
                link_feats.append([topo_spec.tasks[task_id].intra_bandwidth])
                link_feats.append([topo_spec.tasks[task_id].intra_bandwidth])
            else:
                link_feats.append([topo_spec.bandwidth[task_id][another_task_id]])
                link_feats.append([topo_spec.bandwidth[another_task_id][task_id]])
    link_feats = np.array(link_feats, dtype=float)

    nccl_models = gen_nccl_model(topo_spec)
    n_op, n_dev = len(gdef.node), len(devices)

    parameter_sizes = np.zeros(n_op)
    tensor_sizes = np.zeros((n_op, n_op))
    computation_times = np.zeros((n_op, n_dev, 4))

    name_dict = { node.name: i for i, node in enumerate(gdef.node) }
    for thisnodeid, node in enumerate(gdef.node):
        for input in node.input:
            x, input_index = parse_input(input)
            nodeid = name_dict[x]
            tensorsize = get_input_size(gdef.node[nodeid], input_index, batchsize)
            tensor_sizes[(thisnodeid, nodeid)] += tensorsize
            if node.op.startswith('Apply') and input_index == 0: # this input is a variable tensor
                parameter_sizes[nodeid] = tensorsize
        for device_id, (_, task_id) in enumerate(devices):
            pdata = prof_data[topo_spec.tasks[task_id].gpu_model]
            computation_times[thisnodeid, device_id, 0] += pdata[(node.name, 1)][0]
            computation_times[thisnodeid, device_id, 1] += pdata[(node.name, 2)][0]
            computation_times[thisnodeid, device_id, 2] += pdata[(node.name, 4)][0]
            computation_times[thisnodeid, device_id, 3] += pdata[(node.name, 8)][0]

    op_feats = np.array([[np.mean(computation_times[i, :, x]) for x in range(4)] + [parameter_sizes[i], tensor_sizes[i, i]] for i in range(n_op)], dtype=float)
    tensor_feats = []
    place_feats = []
    edge_prev = ([], [])
    edge_succ = ([], [])
    edge_place = ([], [])
    edge_serve = ([], [])

    for i in range(n_op):
        for j in range(n_op):
            if i == j:
                continue
            if tensor_sizes[i, j] > 0:
                edge_prev[0].append(i)
                edge_prev[1].append(j)
                edge_succ[0].append(j)
                edge_succ[1].append(i)
                tensor_feats.append([tensor_sizes[i, j]])

    for op_id in range(n_op):
        for dev_id in range(n_dev):
            edge_place[0].append(op_id)
            edge_place[1].append(dev_id)
            edge_serve[0].append(dev_id)
            edge_serve[1].append(op_id)
            place_feats.append([ computation_times[op_id, dev_id, x] for x in range(4) ])

    tensor_feats = np.array(tensor_feats, dtype=float)
    place_feats = np.array(place_feats, dtype=float)

    prof_data_combined = { key: [] for key in prof_data[topo_spec.tasks[0].gpu_model].keys() }
    for _, task_id in devices:
        for key, times in prof_data[topo_spec.tasks[task_id].gpu_model].items():
            prof_data_combined[key].append(times[0])

    g = dgl.heterograph({
        ('device', 'link', 'device'): edge_link,
        ('op', 'prev', 'op'): edge_prev,
        ('op', 'succ', 'op'): edge_succ,
        ('op', 'place', 'device'): edge_place,
        ('device', 'serve', 'op'): edge_serve
    })

    topology_for_simulator = gen_topology_for_simulator(topo_spec)

    # note for normalization: regard tensor size as the time required to trasfer it in unit (maximum) bandwidth
    # 0. (not implemented) use second as time units and MB as size units?
    # 1. calculate CL2 = ||computation times||_2 and divide all computation time
    # 2. calculate maximum bandwidth Bmax and divide all bandwidth by it
    # 3. divide all tensors and memory limits by CL2*B
    CL2 = np.linalg.norm(op_feats[:, 0])
    Bmax = max( np.max(topo_spec.bandwidth), max(x.intra_bandwidth for x in topo_spec.tasks) )

    op_feats[:, 0:4] /= CL2
    op_feats[:, 4:6] /= CL2 * Bmax
    device_feats[:, 0] /= CL2 * Bmax
    link_feats[:, 0] /= Bmax
    tensor_feats[:, 0] /= CL2 * Bmax
    place_feats[:, 0:4] /= CL2

    return {
        "graph": g,
        "gdef": gdef,
        "prof_data": prof_data_combined,
        "topo_spec": topo_spec,
        "devices": devices,
        "op_feats": op_feats,
        "device_feats": device_feats,
        "tensor_feats": tensor_feats,
        "place_feats": place_feats,
        "link_feats": link_feats,
        "scaler": (CL2, Bmax),
        "batchsize": batchsize,
        "topology_for_simulator": topology_for_simulator,
        "nccl_models": nccl_models
    }

def get_all_data():
    models = []
    for m in ("resnet", "inception", "transformer", "bert"): # "vgg", "transformer", "bert",  "mobilenet", "nasnet"
        agg_prof_data = {}
        gdef, batchsize = None, None
        for gtype in ('1080ti', 'v100'):
            gdef, prof_data, _, batchsize = load("{}_{}.pickle".format(m, gtype))
            agg_prof_data[gtype] = prof_data
        models.append((gdef, agg_prof_data, batchsize))

    topos1 = [TopoSpec([
        TopoSpecTask('1080ti', 6<<30, intra, 2),
        TopoSpecTask('1080ti', 6<<30, intra, 2),
        TopoSpecTask('v100',   8<<30, intra, 4),
    ], [[2810, 2810, 2810],
        [2810, 2810, 2810],
        [2810, 2810, 2810]]) for intra in (8000, 20000)]

    topos2 = [TopoSpec([
        TopoSpecTask('1080ti', 6<<30, intra, 2),
        TopoSpecTask('1080ti', 6<<30, intra, 1),
        TopoSpecTask('v100',   8<<30, intra, 2),
    ], [[2810, 2810, 2810],
        [2810, 2810, 2810],
        [2810, 2810, 2810]]) for intra in (8000, 20000)]

    topos3 = [TopoSpec([
        TopoSpecTask('1080ti', 6<<30, intra, 2),
        TopoSpecTask('1080ti', 6<<30, intra, 2),
        TopoSpecTask('v100',   8<<30, intra, 2),
    ], [[2810, 2810, 400],
        [2810, 2810, 400],
        [400, 400, 2810]]) for intra in (8000, 20000)]

    topos4 = [TopoSpec([
        TopoSpecTask('1080ti', 6<<30, 8000, 4),
        TopoSpecTask('1080ti', 6<<30, 8000, 4),
    ], [[2810,2810],
        [2810,2810]]),
    TopoSpec([
        TopoSpecTask('1080ti', 6<<30, 8000, 1),
        TopoSpecTask('1080ti', 6<<30, 8000, 1),
        TopoSpecTask('1080ti', 6<<30, 8000, 1),
        TopoSpecTask('1080ti', 6<<30, 8000, 1),
        TopoSpecTask('1080ti', 6<<30, 8000, 1),
        TopoSpecTask('1080ti', 6<<30, 8000, 1),
        TopoSpecTask('1080ti', 6<<30, 8000, 1),
        TopoSpecTask('1080ti', 6<<30, 8000, 1),
    ], [[2810] * 8] * 8)]

    return [gen_data(gdef, prof_data, batchsize, topo_spec) for gdef, prof_data, batchsize in models for topo_spec in [topos1[0]]]

def gen_nccl_model(topo_spec: TopoSpec):
    # TGE automatically use only the leader (first device) to determin the nccl model to use when no exact model present

    nccl_models = {}

    for task_id, task in enumerate(topo_spec.tasks):
        nccl_models[device_name(task_id, 0)] = [ x * 2810 / task.intra_bandwidth for x in BASE_NCCL_MODEL ]

    for n_tasks in range(2, topo_spec.ntasks + 1):
        for task_ids in itertools.combinations(range(topo_spec.ntasks), n_tasks):
            leaders = [device_name(task_id, 0) for task_id in task_ids]
            min_bandwidth = min(topo_spec.bandwidth[i][j] for i in task_ids for j in task_ids if i != j)
            nccl_models[','.join(sorted(leaders))] = [ x * 2810 / min_bandwidth for x in BASE_NCCL_MODEL ]

    return nccl_models

def gen_topology_for_simulator(topo_spec: TopoSpec):
    links = []
    paths = []

    inter_link_dict = {}

    for first_task_id, first_task in enumerate(topo_spec.tasks):
        for first_index in range(first_task.number):
            for second_task_id, second_task in enumerate(topo_spec.tasks):
                for second_index in range(second_task.number):
                    if first_task_id == second_task_id:
                        if first_index == second_index:
                            paths.append([])
                        else:
                            link_id = len(links)
                            links.append(first_task.intra_bandwidth)
                            paths.append([link_id])
                    else:
                        if (first_task_id, second_task_id) not in inter_link_dict:
                            link_id = len(links)
                            links.append(topo_spec.bandwidth[first_task_id][second_task_id])
                            inter_link_dict[(first_task_id, second_task_id)] = link_id

                        paths.append([inter_link_dict[(first_task_id, second_task_id)]])

    return links, paths

def device_name(task_id, index):
    return "/job:worker/replica:0/task:{}/device:GPU:{}".format(task_id, index)
