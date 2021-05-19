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
    strategy = state.dump_strategy()

    tge = TGE(record['gdef'], [device for device, _ in record['device_list']], sinks=["Adam"])
    tge.set_strategy(strategy)
    tge.fill_batchsize(record['batchsize'])
    tge.replace_placeholder(record['batchsize'])
    tge.set_topology(*record["topology_for_simulator"])
    tge.set_nccl_model(record["nccl_models"])

    temp_path = tempfile.mktemp()
    time, mem = tge.evaluate(record["prof_data"].to_tge(record["topo_spec"], record['batchsize']), chrome_path=trace, dump_path=temp_path)
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
