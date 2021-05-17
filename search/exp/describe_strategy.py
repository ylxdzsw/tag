import numpy as np
import itertools
import copy
import tge
from dataclasses import dataclass
from typing import Any
from data import TopoSpec, TopoSpecTask, ProfileData, device_name, gen_topology_for_simulator, gen_nccl_model
from grouping import group_with_topk_nodes, group_with_tge_basegroups
from utils import info, load
from metis import metis

def explain(m, actions):
    topo_spec = TopoSpec([
        TopoSpecTask('v100',   12<<30, 8000, 2),
        TopoSpecTask('v100',   12<<30, 8000, 2),
        TopoSpecTask('1080ti', 8<<30, 8000, 2),
    ], [[5000, 2180, 5000],
        [2180, 5000, 5000],
        [5000, 5000, 5000]])

    gdef = load('raw_data/{}/model.pickle'.format(m))
    prof_data = ProfileData(m)
    tge.simplify_graph(gdef, sinks=["Adam"])

    batchsize = prof_data.maximum_batchsize()

    base_groups = group_with_tge_basegroups(gdef)
    groups = group_with_topk_nodes(gdef, base_groups, prof_data, n_groups=60)
    sorted_groups = sorted(groups, key=lambda group: -np.sum([ prof_data.get('1080ti', batchsize)[gdef.node[node_id].name] for node_id in group ])) # largest computation time first

    devices = [(device_name(task_id, index), task_id) for task_id, task in enumerate(topo_spec.tasks) for index in range(task.number)]

    strategy = {}
    for gid, group in enumerate(sorted_groups):
        action = actions[gid] if gid < len(actions) else actions[0]

        placed_devices_mask = [ action[0][tid] for (_, tid) in devices ]
        placed_devices = np.nonzero(placed_devices_mask)[0]

        if action[1] == 0: # PS
            for node_id in group:
                s = [ -1-placed_devices[node_id % len(placed_devices)] ]
                s.extend(placed_devices_mask)
                strategy[gdef.node[node_id].name] = s
        elif action[1] == 1: # NCCL
            s = [ 1 ]
            s.extend(placed_devices_mask)
            for node_id in group:
                strategy[gdef.node[node_id].name] = s
        elif action[1] == 2: # MP
            if len(group) <= len(placed_devices): # we have less nodes than device, metis will complain.
                assignments = [ i for i, _ in enumerate(group) ]
            else:
                _, assignments = metis(gdef, {}, len(placed_devices), group, batchsize)
            for node_id, assignment in zip(group, assignments):
                s = [0] * (1 + len(placed_devices))
                s[assignment+1] = 1
                strategy[gdef.node[node_id].name] = s

    N = len(strategy)
    device_count = [0 for _ in devices]
    communication_count = [0, 0]
    for s in strategy.values():
        if s[0] < 0:
            communication_count[0] += 1
        elif s[0] == 1:
            communication_count[1] += 1
        for i in range(len(s) - 1):
            device_count[i] += s[i+1]

    print(*[ c / N for c in communication_count ])
    print(*[ c / N for c in device_count ])
