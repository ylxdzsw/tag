from dataclasses import dataclass

import dgl
import re
import os
import numpy as np
import pickle
import math
import itertools
import networkx as nx
import tge
from bisect import bisect_left
from utils import groupby, car, cadr, cdr, info, load, parse_input, get_input_size
from metis import metis
from grouping import group_with_topk_nodes, group_with_tge_basegroups, group_with_metis
from utils import load, save, info


@dataclass
class TopoSpec:
    tasks: list
    bandwidth: list # symetric matrix for between-task bandwidth

    @property
    def ntasks(self):
        return len(self.tasks)

    def devices(self):
        return ( (device_name(task_id, i), task.gpu_model, task.memory) for task_id, task in enumerate(self.tasks) for i in range(task.number) )

@dataclass
class TopoSpecTask:
    gpu_model: str
    memory: int
    intra_bandwidth: int
    number: int

BASE_NCCL_MODEL = [0.043420241077615454, 368.2013618677043, 0.27766802543921265, 211.91926070037152] # f: size -> time

def gen_data(gdef, prof_data, batchsize, topo_spec: TopoSpec):
    device_list = [(device_name(task_id, index), task_id) for task_id, task in enumerate(topo_spec.tasks) for index in range(task.number)]
    device_segment = [ task_id for device_name, task_id in device_list ]

    device_feats = np.array([[task.number, task.memory, task.intra_bandwidth] for task in topo_spec.tasks], dtype=float)

    edge_link = [], []
    link_feats = []

    for task_id, _ in enumerate(topo_spec.tasks):
        for another_task_id, _ in enumerate(topo_spec.tasks):
            if another_task_id == task_id:
                continue
            edge_link[0].append(task_id)
            edge_link[1].append(another_task_id)
            edge_link[0].append(another_task_id)
            edge_link[1].append(task_id)

            link_feats.append([topo_spec.bandwidth[task_id][another_task_id]])
            link_feats.append([topo_spec.bandwidth[another_task_id][task_id]])
    link_feats = np.array(link_feats, dtype=float)

    nccl_models = gen_nccl_model(topo_spec)
    n_op, n_dev = len(gdef.node), len(device_list)

    parameter_sizes = np.zeros(n_op)
    tensor_sizes = np.zeros((n_op, n_op))
    computation_times = np.zeros((n_op, n_dev, 4))

    name_dict = { node.name: i for i, node in enumerate(gdef.node) }
    for thisnodeid, node in enumerate(gdef.node):
        for input in node.input:
            x, input_index = parse_input(input)
            if x not in name_dict:
                info(x)
            nodeid = name_dict[x]
            tensorsize = get_input_size(gdef.node[nodeid], input_index, batchsize)
            tensor_sizes[nodeid, thisnodeid] += tensorsize
            if node.op.startswith('Apply') and input_index == 0: # this input is a variable tensor
                parameter_sizes[nodeid] = tensorsize
        for device_id, (_, task_id) in enumerate(device_list):
            computation_times[thisnodeid, device_id, 0] += prof_data.get(topo_spec.tasks[task_id].gpu_model, batchsize)[node.name]
            computation_times[thisnodeid, device_id, 1] += prof_data.get(topo_spec.tasks[task_id].gpu_model, batchsize // 2)[node.name]
            computation_times[thisnodeid, device_id, 2] += prof_data.get(topo_spec.tasks[task_id].gpu_model, batchsize // 4)[node.name]
            computation_times[thisnodeid, device_id, 3] += prof_data.get(topo_spec.tasks[task_id].gpu_model, batchsize // 8)[node.name]

    base_groups = group_with_tge_basegroups(gdef)
    costs = [ int(x) for x in np.sqrt(np.mean(computation_times, (1,2))) ] #  + np.sqrt(parameter_sizes)
    op_groups = group_with_metis(gdef, base_groups, costs, batchsize, n_groups=40)
    # op_groups = group_with_topk_nodes(gdef, base_groups, prof_data, n_groups=40)
    op_groups = sorted(op_groups, key=lambda group: -np.sum([ np.average(computation_times[node_id, :, 0]) for node_id in group ])) # largest computation time first
    op_segment = [0] * len(gdef.node)
    for group_id, ops in enumerate(op_groups):
        for node_id in ops:
            op_segment[node_id] = group_id

    op_feats = np.array([
        [ np.sum([np.mean(computation_times[op, :, x]) for op in ops]) for x in range(4) ] +
        [ np.sum(parameter_sizes[ops]) ] +
        [ np.sum([ tensor_sizes[op1, op2] for op1 in ops for op2 in ops if op1 != op2 ]) ]
        for i, ops in enumerate(op_groups)
    ], dtype=float)

    tensor_dict = {}
    tensor_feats = []
    edge_prev = ([], [])
    edge_succ = ([], [])

    for i in range(n_op):
        for j in range(n_op):
            if i == j:
                continue
            if tensor_sizes[i, j] <= 0:
                continue

            gi, gj = op_segment[i], op_segment[j]
            if (gi, gj) in tensor_dict:
                tensor_id = tensor_dict[(gi, gj)]
            else:
                tensor_id = len(tensor_dict)
                tensor_dict[(gi, gj)] = tensor_id
                edge_prev[0].append(gi)
                edge_prev[1].append(gj)
                edge_succ[0].append(gj)
                edge_succ[1].append(gi)
                tensor_feats.append([0])

            tensor_feats[tensor_id][0] == tensor_sizes[i, j]

    place_dict = {}
    place_feats = []
    edge_place = ([], [])
    edge_serve = ([], [])

    for gid in range(len(op_groups)):
        for tid in range(topo_spec.ntasks):
            place_id = len(place_dict)
            place_dict[(gid, tid)] = place_id
            edge_place[0].append(gid)
            edge_place[1].append(tid)
            edge_serve[0].append(tid)
            edge_serve[1].append(gid)
            place_feats.append([ 0 ] * 4)

    for op_id in range(n_op):
        for dev_id in range(n_dev):
            gid, tid = op_segment[op_id], device_segment[dev_id]
            place_id = place_dict[(gid, tid)]

            for x in range(4):
                place_feats[place_id][x] += computation_times[op_id, dev_id, x]

    tensor_feats = np.array(tensor_feats, dtype=float)
    place_feats = np.array(place_feats, dtype=float)

    topology_for_simulator = gen_topology_for_simulator(topo_spec)

    # note for normalization: regard tensor size as the time required to trasfer it in unit (maximum) bandwidth
    # 0. (not implemented) use second as time units and MB as size units?
    # 1. calculate CL2 = ||computation times||_2 and divide all computation time
    # 2. calculate maximum bandwidth Bmax and divide all bandwidth by it
    # 3. divide all tensors and memory limits by CL2*Bmax
    CL2 = np.linalg.norm(op_feats[:, 0])
    Bmax = max( np.max(topo_spec.bandwidth), max(x.intra_bandwidth for x in topo_spec.tasks) )

    op_feats[:, 0:4] /= CL2
    op_feats[:, 4:6] /= CL2 * Bmax
    device_feats[:, 0] /= 4 # the maximum number of GPUs in a task
    device_feats[:, 1] /= CL2 * Bmax
    device_feats[:, 2] /= Bmax
    link_feats[:, 0] /= Bmax
    tensor_feats[:, 0] /= CL2 * Bmax
    place_feats[:, 0:4] /= CL2

    g = dgl.heterograph({
        ('device', 'link', 'device'): edge_link,
        ('op', 'prev', 'op'): edge_prev,
        ('op', 'succ', 'op'): edge_succ,
        ('op', 'place', 'device'): edge_place,
        ('device', 'serve', 'op'): edge_serve
    })

    return {
        "graph": g,
        "gdef": gdef,
        "prof_data": prof_data,
        "topo_spec": topo_spec,

        "op_groups": op_groups, # sorted with largest computation cost first
        "op_segments": op_segment,
        "op_feats": op_feats,

        "device_list": device_list,
        "device_segments": device_segment, # sorted by task id
        "device_feats": device_feats,

        "tensor_feats": tensor_feats,
        "tensor_dict": tensor_dict,

        "place_feats": place_feats,
        "place_dict": place_dict,

        "link_feats": link_feats,

        "parameter_sizes": parameter_sizes,
        "tensor_sizes": tensor_sizes,

        "scaler": (CL2, Bmax),
        "batchsize": batchsize,
        "topology_for_simulator": topology_for_simulator,
        "nccl_models": nccl_models
    }

def get_all_data():
    records = []

    real_topo = TopoSpec([
        TopoSpecTask('v100', 30<<30, 8000, 4),
        TopoSpecTask('1080ti', 9<<30, 3000, 8),
        TopoSpecTask('p100', 10<<30, 3000, 4),
    ], [[2810 for _ in range(3)] for _ in range(3)])

    for m in ("inception", "resnet", "vgg", "transformer", "bert", "berts"): # (, "bert", "berts" "rnnlm2x", "rnnlm4x"): #  , "mobilenet", "nasnet"
        gdef = load('raw_data/{}/model.pickle'.format(m))
        prof_data = ProfileData(m)
        tge.simplify_graph(gdef, sinks=["Adam"])

        model_size = estimate_model_size(gdef, prof_data.maximum_batchsize())
        for i in range(8):
            info("generating {} topo {}".format(m, i))
            topo = gen_random_topology(model_size)
            record = gen_data(gdef, prof_data, prof_data.maximum_batchsize(), topo)
            record['model_name'] = m
            record['topo_name'] = i
            records.append(record)
        info("generating {} real topo".format(m))
        record = gen_data(gdef, prof_data,  prof_data.maximum_batchsize(), real_topo)
        record['model_name'] = m
        record['topo_name'] = 'real'
        records.append(record)

    for i, record in enumerate(records):
        record['id'] = i

    return records

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

def gen_random_topology(model_size):
    total_memory = 0

    gpu_models = ('p100', '1080ti', 'v100')
    gpu_memory = {
        'p100': 10<<30,
        '1080ti': 9<<30,
        'v100': 14<<30
    }
    intra_links = (8000, 50000) # PCI, nvlink
    inter_links = (2810, 5000) # diff rack, same rack
    card_numbers = (2, 4)

    tasks = []
    for _ in range(6): # at most 6 tasks
        if len(tasks) >= 2 and total_memory > 2 * model_size and np.random.rand() < .5:
            break

        gpu_model = np.random.choice(gpu_models)
        memory = gpu_memory[gpu_model]
        card_number = np.random.choice(card_numbers)
        intra_bandwidth = np.random.choice(intra_links)
        task = TopoSpecTask(gpu_model, memory, intra_bandwidth, card_number)

        total_memory += card_number * memory
        tasks.append(task)

    n = len(tasks)
    racks = [ np.random.randint(n) for _ in range(n) ]
    inter_bandwidth = [ [0 for _ in range(n)] for _ in range(n) ]
    for i in range(n):
        for j in range(i+1):
            bandwidth = inter_links[0] if racks[i] != racks[j] else inter_links[1]
            inter_bandwidth[i][j] = bandwidth
            inter_bandwidth[j][i] = bandwidth

    return TopoSpec(tasks, inter_bandwidth)

def estimate_model_size(gdef, batchsize):
    parameter_sizes = [0 for _ in range(len(gdef.node))]
    name_dict = { node.name: i for i, node in enumerate(gdef.node) }
    for node in gdef.node:
        for input in node.input:
            x, input_index = parse_input(input)
            if node.op.startswith('Apply') and input_index == 0: # this input is a variable tensor
                nodeid = name_dict[x]
                parameter_sizes[nodeid] = get_input_size(gdef.node[nodeid], input_index, batchsize)
    return sum(parameter_sizes) * 4

class ProfileData:
    ALL_GTYPES = ('p100', '1080ti', 'v100')

    def __init__(self, model_name):
        self.data = {}
        for gtype in ProfileData.ALL_GTYPES:
            files = os.listdir('raw_data/{}/{}'.format(model_name, gtype))
            self.batch_sizes = sorted([ int(x.split('.')[0]) for x in files ])
            self.data[gtype] = {}
            for b in self.batch_sizes:
                self.data[gtype][b] = load('raw_data/{}/{}/{}.pickle'.format(model_name, gtype, b))

    def maximum_batchsize(self):
        return self.batch_sizes[-1]

    # use the 2 nearest points to fit a linear model
    def linear_fit(self, target):
        for gtype in self.data:
            i = bisect_left(self.batch_sizes, target)
            if i >= len(self.batch_sizes):
                i -= 1
            elif i == 0:
                i += 1
            x1, x2 = self.batch_sizes[i-1], self.batch_sizes[i]
            fitted = {}
            for key in self.data[gtype][x1]:
                y1, y2 = self.data[gtype][x1][key], self.data[gtype][x2][key]
                predicted = y2 * (target - x1) / (x2 - x1) + y1 * (x2 - target) / (x2 - x1)
                fitted[key] = int(max(predicted, 0))
            self.data[gtype][target] = fitted

    def get(self, gtype, batch_size):
        if batch_size not in self.data[gtype]:
            self.linear_fit(batch_size)
        return self.data[gtype][batch_size]

    def to_tge_single(self, gtype, target, nrep):
        result = {}
        for i in range(1, nrep+1):
            if target % i == 0:
                p = self.get(gtype, target // i)
                for x in p:
                    result[(x, i)] = p[x]
        return result

    def to_tge(self, topo_spec, batch_size):
        gtypes = [ gtype for name, gtype, memory in topo_spec.devices() ]
        nrep = len(gtypes)
        cache = { gtype: self.to_tge_single(gtype, batch_size, nrep) for gtype in ProfileData.ALL_GTYPES }
        result = {}
        for key in cache[gtypes[0]]:
            result[key] = [ cache[gtype][key] for gtype in gtypes ]

        return result
