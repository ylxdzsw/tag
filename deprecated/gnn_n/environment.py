from tge import TGE
from utils import car, cadr, cdr, info
from metis import metis
import numpy as np
import tensorflow as tf

def sample(logit, e=0):
    p = tf.math.sigmoid(logit)
    def f(x):
        if np.random.rand() < e:
            return np.random.choice(2)
        else:
            return int(np.random.rand() < x)
    return np.vectorize(f)(p)

def evaluate(record, nodemask, ncclmask, psmask):
    gdef = record["gdef"]
    # replication_number_feasibility_rounding(record, nodemask)
    strategy = {}
    for gi, group in enumerate(record["op_groups"]):
        if ncclmask[gi] == 2: # metis model parallel
            device_ids = []
            i = 0
            for task_id, task in enumerate(record["topo_spec"].tasks):
                for index in range(task.number):
                    if int(nodemask[gi, task_id]) > 0:
                        device_ids.append(i)
                    i += 1
            _, assignments = metis(record["gdef"], {}, len(device_ids), group, record["batchsize"])
            for node_id, assignment in zip(group, assignments):
                s = [0] * (1 + len(record["devices"]))
                s[assignment+1] = 1
                strategy[gdef.node[node_id].name] = s
        else:
            for node_id in group:
                s = [-1-int(psmask[gi]) if int(ncclmask[gi]) == 0 else int(ncclmask[gi])]
                for task_id, task in enumerate(record["topo_spec"].tasks):
                    for index in range(task.number):
                        s.append(int(nodemask[gi, task_id]))
                strategy[gdef.node[node_id].name] = s
    # info(strategy)
    leftout = [ gi for gi in range(len(record["op_groups"])) if np.sum(nodemask[gi, :]) == 0 ]
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            v[1] = 1
    tge = TGE(gdef, [device for device in record["devices"]], sinks=["Adam"])
    tge.set_strategy(strategy)
    tge.fill_batchsize(record["batchsize"])
    tge.replace_placeholder(record["batchsize"])
    tge.set_topology(*record["topology_for_simulator"])
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"])

    oom, i = [], 0
    for task_id, task in enumerate(record["topo_spec"].tasks):
        for index in range(task.number):
            if mem[i] > task.memory:
                oom.append(i)
            i = i + 1
    return np.sqrt(time / 1_000_000), oom, leftout

def score(time, oom, leftout):
    nerror = len(oom) + len(leftout)
    return time * (1 + 10 * nerror)

def base_strategies(record):
    result = []

    ncgroups = len(record['op_groups'])
    ntasks = record['topo_spec'].ntasks

    # 1: task0
    s = np.zeros((ncgroups, ntasks), dtype=np.int)
    for i in range(ncgroups):
        s[i, 0] = 1
    result.append((s, [0] * ncgroups, [0] * ncgroups))

    # 2: task0 + task1 + nccl
    s = np.zeros((ncgroups, ntasks), dtype=np.int)
    for i in range(ncgroups):
        s[i, 0] = 1
        s[i, 1] = 1
    result.append((s, [1] * ncgroups, [0] * ncgroups))

    # 3: task0 + task1 + task2 + task3 + nccl
    if ntasks >= 4:
        s = np.zeros((ncgroups, ntasks), dtype=np.int)
        for i in range(ncgroups):
            s[i, 0] = 1
            s[i, 1] = 1
            s[i, 2] = 1
            s[i, 3] = 1
        result.append((s, [1] * ncgroups, [0] * ncgroups))

    # 4. all + ps (round robin)
    s = np.ones((ncgroups, ntasks), dtype=np.int)
    result.append((s, [0] * ncgroups, [ i % ntasks for i in range(ncgroups) ]))

    # 5. all + nccl
    s = np.ones((ncgroups, ntasks), dtype=np.int)
    result.append((s, [1] * ncgroups, [0] * ncgroups))

    # 6. all + metis
    s = np.ones((ncgroups, ntasks), dtype=np.int)
    result.append((s, [2] * ncgroups, [0] * ncgroups))

    return result

def replication_number_feasibility_rounding(record, nodemask):
    B = record["batchsize"]

    for i in range(nodemask.shape[0]):
        r = sum(nodemask[i, :])
        if B % r == 0:
            continue

        actions = []
        if r > 1 and B % (r - 1) == 0: # try remove one replica. Devices that have two replicas has double probability
            for j, k in enumerate(nodemask[i, :]):
                actions.extend([(j, -1)] * k)
        if B % (r + 1) == 0: # try add one replica, but only on devices that already have one
            for j, k in enumerate(nodemask[i, :]):
                if k == 1:
                    actions.append((j, 1))

        if len(actions) < 0: # heuristic failed, randomly remove replicas until feasible
            while B % sum(nodemask[i, :]) != 0:
                j = np.random.chioce(nodemask.shape[1])
                if nodemask[i, j] > 0:
                    nodemask[i, j] -= 1
            continue

        j, a = np.random.choice(actions)
        nodemask[i, j] += a

    return nodemask

def fitness(arg):
    record, pheno = arg
    nodemask = np.reshape(pheno[:len(record['op_groups']) * record['topo_spec'].ntasks], (len(record['op_groups']), record['topo_spec'].ntasks))
    ncclmask = (pheno[len(record['op_groups']) * record['topo_spec'].ntasks:] == 0).astype(int)
    psmask = pheno[len(record['op_groups']) * record['topo_spec'].ntasks:] - 1

    return score(*evaluate(record, nodemask, ncclmask, psmask))

def quick_fitness(arg):
    record, pheno = arg
    nodemask = np.reshape(pheno, (len(record['op_groups']), record['topo_spec'].ntasks))
    ncclmask = [1] * len(record['op_groups'])
    psmask = [0] * len(record['op_groups'])

    return score(*evaluate(record, nodemask, ncclmask, psmask))
