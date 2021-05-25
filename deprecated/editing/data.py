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
from grouping import group_with_topk_nodes, group_with_tge_basegroups

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
            if x not in name_dict:
                info(x)
            nodeid = name_dict[x]
            tensorsize = get_input_size(gdef.node[nodeid], input_index, batchsize)
            tensor_sizes[(thisnodeid, nodeid)] += tensorsize
            if node.op.startswith('Apply') and input_index == 0: # this input is a variable tensor
                parameter_sizes[nodeid] = tensorsize
        for device_id, (_, task_id) in enumerate(devices):
            computation_times[thisnodeid, device_id, 0] += prof_data.get(topo_spec.tasks[task_id].gpu_model, batchsize)[node.name]
            computation_times[thisnodeid, device_id, 1] += prof_data.get(topo_spec.tasks[task_id].gpu_model, batchsize // 2)[node.name]
            computation_times[thisnodeid, device_id, 2] += prof_data.get(topo_spec.tasks[task_id].gpu_model, batchsize // 4)[node.name]
            computation_times[thisnodeid, device_id, 3] += prof_data.get(topo_spec.tasks[task_id].gpu_model, batchsize // 8)[node.name]

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

    g_full = dgl.heterograph({
        ('device', 'link', 'device'): edge_link,
        ('op', 'prev', 'op'): edge_prev,
        ('op', 'succ', 'op'): edge_succ,
        ('op', 'place', 'device'): edge_place,
        ('device', 'serve', 'op'): edge_serve
    })

    base_groups = group_with_tge_basegroups(gdef)
    groups = group_with_topk_nodes(gdef, base_groups, prof_data, n_groups=20)

    op_segment = [0] * op_feats.shape[0]
    for group_id, group in enumerate(groups):
        for node_id in group:
            op_segment[node_id] = group_id

    edge_prev_group_dict = {}
    edge_prev_segment = [] # the segment id
    grouped_edge_prev = [], []
    for i, (a, b) in enumerate(zip(*edge_prev)):
        ga, gb = op_segment[a], op_segment[b]
        if (ga, gb) in edge_prev_group_dict:
            edge_prev_segment.append(edge_prev_group_dict[(ga, gb)])
        else:
            group_id = len(edge_prev_group_dict)
            edge_prev_group_dict[(ga, gb)] = group_id
            grouped_edge_prev[0].append(ga)
            grouped_edge_prev[1].append(gb)
            edge_prev_segment.append(group_id)

    edge_succ_group_dict = {}
    edge_succ_segment = []
    grouped_edge_succ = [], []
    for i, (a, b) in enumerate(zip(*edge_succ)):
        ga, gb = op_segment[a], op_segment[b]
        if (ga, gb) in edge_succ_group_dict:
            edge_succ_segment.append(edge_succ_group_dict[(ga, gb)])
        else:
            group_id = len(edge_succ_group_dict)
            edge_succ_group_dict[(ga, gb)] = group_id
            grouped_edge_succ[0].append(ga)
            grouped_edge_succ[1].append(gb)
            edge_succ_segment.append(group_id)

    edge_place_group_dict = {}
    edge_place_segment = []
    grouped_edge_place = [], []
    for i, (a, b) in enumerate(zip(*edge_place)):
        ga = op_segment[a]
        if (ga, b) in edge_place_group_dict:
            edge_place_segment.append(edge_place_group_dict[(ga, b)])
        else:
            group_id = len(edge_place_group_dict)
            edge_place_group_dict[(ga, b)] = group_id
            grouped_edge_place[0].append(ga)
            grouped_edge_place[1].append(b)
            edge_place_segment.append(group_id)

    edge_serve_group_dict = {}
    edge_serve_segment = []
    grouped_edge_serve = [], []
    for i, (a, b) in enumerate(zip(*edge_serve)):
        gb = op_segment[b]
        if (a, gb) in edge_serve_group_dict:
            edge_serve_segment.append(edge_serve_group_dict[(a, gb)])
        else:
            group_id = len(edge_serve_group_dict)
            edge_serve_group_dict[(a, gb)] = group_id
            grouped_edge_serve[0].append(a)
            grouped_edge_serve[1].append(gb)
            edge_serve_segment.append(group_id)

    g_grouped = dgl.heterograph({
        ('device', 'link', 'device'): edge_link,
        ('op', 'prev', 'op'): grouped_edge_prev,
        ('op', 'succ', 'op'): grouped_edge_succ,
        ('op', 'place', 'device'): grouped_edge_place,
        ('device', 'serve', 'op'): grouped_edge_serve
    })

    return {
        "graph": g_grouped,
        "graph_full": g_full,
        "gdef": gdef,
        "prof_data": prof_data,
        "topo_spec": topo_spec,
        "devices": devices,
        "groups": groups,
        "segments": {
            "op": (op_segment, len(groups)),
            "prev": (edge_prev_segment, len(edge_prev_group_dict)),
            "succ": (edge_succ_segment, len(edge_succ_group_dict)),
            "place": (edge_place_segment, len(edge_place_group_dict)),
            "serve": (edge_serve_segment, len(edge_serve_group_dict))
        },
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
    records = []

    real_topo = TopoSpec([
        TopoSpecTask('1080ti', 6<<30, 5000, 2),
        TopoSpecTask('1080ti', 6<<30, 5000, 2),
        TopoSpecTask('v100',   8<<30, 5000, 4),
    ], [[2810, 2810, 2810],
        [2810, 2810, 2810],
        [2810, 2810, 2810]])

    for m in ("inception", "transformer"): # ("inception", "resnet", "vgg", "transformer", "bert", "rnnlm2x", "rnnlm4x"): #  , "mobilenet", "nasnet"
        gdef = load('raw_data/{}/model.pickle'.format(m))
        prof_data = ProfileData(m)
        tge.simplify_graph(gdef, sinks=["Adam"])

        model_size = estimate_model_size(gdef, prof_data.maximum_batchsize())
        for _ in range(16):
            topo = gen_random_topology(model_size)
            records.append(gen_data(gdef, prof_data, prof_data.maximum_batchsize(), topo))
        records.append(gen_data(gdef, prof_data,  prof_data.maximum_batchsize(), real_topo))

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

    gpu_models = ('1080ti', 'v100')
    gpu_memory = {
        '1080ti': 6<<28,#6<<30,
        'v100': 8<<28,#8<<30
    }
    intra_links = (5000, 50000) # PCI, nvlink
    inter_links = (2810, 8000, 25000)
    card_numbers = (1, 2, 4)

    tasks = []
    for _ in range(8): # at most 8 tasks
        if sum(t.number for t in tasks) >= 2 and total_memory > model_size*1.5 and np.random.rand() < .5:
            break

        gpu_model = np.random.choice(gpu_models)
        memory = gpu_memory[gpu_model]
        card_number = np.random.choice(card_numbers)
        intra_bandwidth = np.random.choice(intra_links)
        task = TopoSpecTask(gpu_model, memory, intra_bandwidth, card_number)

        total_memory += card_number * memory
        tasks.append(task)

    n = len(tasks)
    inter_bandwidth = [ [0 for _ in range(n)] for _ in range(n) ]
    for i in range(n):
        for j in range(i+1):
            bandwidth = np.random.choice(inter_links)
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
    ALL_GTYPES = ('1080ti', 'v100')

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
