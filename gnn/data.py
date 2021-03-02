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

MEMORY_NORMALIZATION: int = 1 / 10_000_000_000
BANDWIDTH_NORMALIZATION: int = 1 / 100_000
BASE_NCCL_MODEL = [0.043420241077615454, 368.2013618677043, 0.27766802543921265, 211.91926070037152] # f: size -> time

def gen_data(gdef, prof_data, batchsize, topo_spec: TopoSpec):
    edge_link = [], []
    link_feats = []
    task_feats = [[task.memory * MEMORY_NORMALIZATION, task.intra_bandwidth * BANDWIDTH_NORMALIZATION, task.number] for task in topo_spec.tasks]

    for task_id, task in enumerate(topo_spec.tasks):
        for another_task_id, another_task in enumerate(topo_spec.tasks):
            if another_task_id == task_id:
                continue
            edge_link[0].append(task_id)
            edge_link[1].append(another_task_id)
            link_feats.append([topo_spec.bandwidth[task_id][another_task_id] * BANDWIDTH_NORMALIZATION])

            edge_link[0].append(another_task_id)
            edge_link[1].append(task_id)
            link_feats.append([topo_spec.bandwidth[another_task_id][task_id] * BANDWIDTH_NORMALIZATION])

    nccl_models = gen_nccl_model(topo_spec)

    devices = [device_name(task_id, index) for task_id, task in enumerate(topo_spec.tasks) for index in range(task.number)]

    def group_with_topk_layers(n_groups=20):
        # NOTE: the format of basegroups is like [0, 1,2,3,4,3,2]. i.e. the i-th element is the group id of i-th node
        # group_table = {}
        # for i, node in enumerate(gdef.node):
        #     if node.name.startswith("GradientDescent") or node.name.startswith("gradients"):
        #         prefix = '/'.join(node.name.split('/')[1:3])
        #     else:
        #         prefix = '/'.join(node.name.split('/')[:2])
        #     if prefix in group_table:
        #         group_table[prefix].append(i)
        #     else:
        #         group_table[prefix] = [i]
        # return list(group_table.values())

        from utils import group_around_topk_costs
        from tge import TGE

        base_groups = TGE(gdef, devices).get_groups()
        id_list = group_around_topk_costs(gdef, base_groups, lambda x: prof_data[topo_spec.tasks[0].gpu_model][(x, 1)][0], n_groups-1) # TODO: use average time in all gpu types? weighted average?
        return list(groupby(enumerate(id_list), key=cadr, value=car).values())

    def group_with_metis(n_groups=20):
        from tge import TGE

        base_groups = TGE(gdef, devices).get_groups()
        _, id_list = metis(gdef, prof_data[topo_spec.tasks[0].gpu_model], n_groups, list(range(len(gdef.node))), batchsize) # TODO: use average time in all gpu types? weighted average?
        return list(groupby(enumerate(id_list), key=cadr, value=car).values())

    n_groups = 4 * topo_spec.ntasks + 10
    # op_groups = group_with_topk_layers(n_groups)
    op_groups = group_with_metis(n_groups)

    parameter_sizes = np.zeros(n_groups)
    tensor_sizes = np.zeros((n_groups, n_groups))
    computation_times = np.zeros((n_groups, topo_spec.ntasks, 4))

    name_dict = { node.name: i for i, node in enumerate(gdef.node) }
    group_dict = { nodeid: groupid for groupid, nodes in enumerate(op_groups) for nodeid in nodes }
    for thisnodeid, node in enumerate(gdef.node):
        thisgroupid = group_dict[thisnodeid]
        for input in node.input:
            x, input_index = parse_input(input)
            nodeid = name_dict[x]
            groupid = group_dict[nodeid]
            tensorsize = get_input_size(gdef.node[nodeid], input_index, batchsize)
            tensor_sizes[(thisgroupid, groupid)] += tensorsize / 100_000_000
            if node.op.startswith('Apply') and input_index == 0: # this input is a variable tensor
                parameter_sizes[groupid] += tensorsize / 100_000_000
        for task_id, task in enumerate(topo_spec.tasks):
            computation_times[thisgroupid, task_id, 0] += prof_data[task.gpu_model][(node.name, 1)][0] / 10_000
            computation_times[thisgroupid, task_id, 1] += prof_data[task.gpu_model][(node.name, 2)][0] / 10_000
            computation_times[thisgroupid, task_id, 2] += prof_data[task.gpu_model][(node.name, 4)][0] / 10_000
            computation_times[thisgroupid, task_id, 3] += prof_data[task.gpu_model][(node.name, 8)][0] / 10_000

    op_feats = [[np.mean(computation_times[i, :, x]) for x in range(4)] + [parameter_sizes[i], tensor_sizes[i, i]] for i in range(n_groups)]
    tensor_feats = []
    place_feats = []
    edge_prev = ([], [])
    edge_succ = ([], [])
    edge_place = ([], [])
    edge_serve = ([], [])

    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                continue
            if tensor_sizes[i, j] > 0:
                edge_prev[0].append(i)
                edge_prev[1].append(j)
                edge_succ[0].append(j)
                edge_succ[1].append(i)
                tensor_feats.append([tensor_sizes[i, j]])

    for op_id in range(n_groups):
        for task_id in range(topo_spec.ntasks):
            edge_place[0].append(op_id)
            edge_place[1].append(task_id)
            edge_serve[0].append(task_id)
            edge_serve[1].append(op_id)
            place_feats.append([ computation_times[op_id, task_id, x] for x in range(4) ])

    prof_data_combined = { key: [] for key in prof_data[topo_spec.tasks[0].gpu_model].keys() }
    for task_id, task in enumerate(topo_spec.tasks):
        for index in range(task.number):
            for key, times in prof_data[task.gpu_model].items():
                prof_data_combined[key].append(times[0])

    g = dgl.heterograph({
        ('task', 'link', 'task'): edge_link,
        ('op', 'prev', 'op'): edge_prev,
        ('op', 'succ', 'op'): edge_succ,
        ('op', 'place', 'task'): edge_place,
        ('task', 'serve', 'op'): edge_serve
    })

    topology_for_simulator = gen_topology_for_simulator(topo_spec)


    return {
        "graph": g,
        "gdef": gdef,
        "prof_data": prof_data_combined,
        "topo_spec": topo_spec,
        "devices": devices,
        "op_groups": op_groups,
        "op_feats": op_feats,
        "task_feats": task_feats,
        "tensor_feats": tensor_feats,
        "place_feats": place_feats,
        "link_feats": link_feats,
        "batchsize": batchsize,
        "topology_for_simulator": topology_for_simulator,
        "nccl_models": nccl_models
    }

def get_all_data():
    models = []
    for m in ("resnet", "inception", "transformer", "bert"): # "vgg", ,  "mobilenet", "nasnet"
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
        [2810, 2810, 2810]]) for intra in (4000, 10000)]

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
