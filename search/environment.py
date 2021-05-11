from tge import TGE
from utils import car, cadr, cdr, info
from metis import metis
import numpy as np
import tensorflow as tf
import tempfile
import os
import json

def evaluate_with_feedback(state, trace=""):
    record = state.record
    gdef = record['gdef']
    devices = record['device_list']
    batchsize = record['batchsize']

    strategy = {}
    for gid, group in enumerate(state.sorted_groups):
        action = state.actions[gid] if gid < len(state.actions) else state.actions[0]

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

    tge = TGE(gdef, [device for device, _ in devices], sinks=["Adam"])
    tge.set_strategy(strategy)
    tge.fill_batchsize(batchsize)
    tge.replace_placeholder(batchsize)
    tge.set_topology(*record["topology_for_simulator"])
    tge.set_nccl_model(record["nccl_models"])

    temp_path = tempfile.mktemp()
    time, mem = tge.evaluate(record["prof_data"].to_tge(record["topo_spec"], batchsize), chrome_path=trace, dump_path=temp_path)
    feedback = json.load(open(temp_path, "r"))
    feedback["peak_memory"] = mem
    os.remove(temp_path)

    return time, feedback

def invalidity(record, feedback): # 0 means valid
    oom = 0
    for peak_memory, (_, task_id) in zip(feedback["peak_memory"], record["device_list"]):
        if peak_memory > record["topo_spec"].tasks[task_id].memory:
            oom += 1
    return oom / len(record["device_list"])
