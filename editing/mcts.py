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

@dataclass
class Action:
    placement: Any # a list of the same length of machines
    communication: Any # 0: PS, 1: NCCL, 2: MP

@dataclass
class State:
    gdef: Any
    batchsize: Any
    topo_spec: Any
    prof_data: Any
    sorted_groups: Any

    topology_for_simulator: Any
    nccl_model: Any
    prof_data_for_simulator: Any
    dp_time: Any

    actions: Any # the actions taken so far. The rest nodes uses the first action (same strategy as the most computational expensive group)

    # shallow copy except for the actions
    def clone(self):
        x = copy.copy(self)
        x.actions = copy.deepcopy(self.actions)
        return x

class Node:
    def __init__(self, action):
        self.action = action
        self.p = 0
        self.q = 0
        self.n_visits = 0
        self.children = []
        self.value = None

    def playout_and_update_recursive(self, state):
        if self.is_leaf():
            if len(state.actions) < len(state.sorted_groups):
                self.expand(state)
            if len(state.actions) == 0: # root at first
                return 0.
            leaf_value = self.evaluate(state)
            self.update(leaf_value)

            return leaf_value
        child = self.select_child()
        state.actions.append(child.action)
        leaf_value = child.playout_and_update_recursive(state)
        self.update(leaf_value)
        return leaf_value

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        return max(self.children, key=lambda x: x.puct(self.n_visits))

    def puct(self, pvisit):
        return self.q + 1.4 * self.p * np.sqrt(pvisit) / (1 + self.n_visits)

    def update(self, leaf_value):
        self.n_visits += 1
        self.q += (leaf_value - self.q) / self.n_visits

    def expand(self, state):
        for placement in itertools.product([0, 1], repeat=len(state.topo_spec.tasks)):
            if sum(placement) == 0:
                continue

            for communication in range(3):
                if communication == 1 and sum(placement) == 1:
                    continue

                action = placement, communication
                if len(state.actions) > 0 and action == state.actions[0]:
                    continue

                self.children.append(Node(action))

        for child in self.children:
            child.p = 1 / len(self.children)

    def evaluate(self, state):
        if self.value is None:
            time, oom = simulate(state)
            speed_up = -1 if oom > 0 else state.dp_time / time - 1
            self.value = speed_up
        return self.value

class Tree:
    def __init__(self):
        self.root = Node(None)

    def playout(self, state, ntimes):
        best = -1
        best_actions = None
        for n in range(ntimes):
            state_clone = state.clone()
            leaf_value = self.root.playout_and_update_recursive(state_clone)
            if leaf_value > best:
                best = leaf_value
                best_actions = state_clone.actions
            info(leaf_value, state_clone.actions)
        return best, best_actions

    def get_action(self):
        return max(self.root.children, key=lambda x: x.n_visits).action

def search(gdef, topo_spec, prof_data):
    batchsize = prof_data.maximum_batchsize()

    base_groups = group_with_tge_basegroups(gdef)
    groups = group_with_topk_nodes(gdef, base_groups, prof_data, n_groups=60)
    sorted_groups = sorted(groups, key=lambda group: -np.sum([ prof_data.get('1080ti', batchsize)[gdef.node[node_id].name] for node_id in group ])) # largest computation time first

    state = State(
        gdef,
        batchsize,
        topo_spec,
        prof_data,
        sorted_groups,

        gen_topology_for_simulator(topo_spec), # topology_for_simulator
        gen_nccl_model(topo_spec), # nccl_model
        prof_data.to_tge(topo_spec, batchsize), # prof_data_for_simulator
        0, # dp_time

        [] #actions
    )

    state_copy = state.clone()
    state_copy.actions.append(([1 for _ in range(len(topo_spec.tasks))], 1))
    time, _ = simulate(state_copy)

    # TODO: if OOM, use MP as baseline
    state.dp_time = time

    best_score, best_actions = Tree().playout(state, 50000)
    return best_score, best_actions

def simulate(state):
    devices = [(device_name(task_id, index), task_id) for task_id, task in enumerate(state.topo_spec.tasks) for index in range(task.number)]

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
                _, assignments = metis(state.gdef, {}, len(placed_devices), group, state.batchsize)
            for node_id, assignment in zip(group, assignments):
                s = [0] * (1 + len(placed_devices))
                s[assignment+1] = 1
                strategy[gdef.node[node_id].name] = s

    t = tge.TGE(state.gdef, [device for device, _ in devices], sinks=["Adam"])
    t.set_strategy(strategy)
    t.fill_batchsize(state.batchsize)
    t.replace_placeholder(state.batchsize)
    t.set_topology(*state.topology_for_simulator)
    t.set_nccl_model(state.nccl_model)

    time, mem = t.evaluate(state.prof_data_for_simulator)
    oom = 0
    for peak_memory, (_, task_id) in zip(mem, devices):
        if peak_memory > state.topo_spec.tasks[task_id].memory:
            oom += 1

    return time, oom

if __name__ == '__main__':
    import sys

    m = sys.argv[1]

    topo = TopoSpec([
        TopoSpecTask('v100',   12<<30, 8000, 2),
        TopoSpecTask('v100',   12<<30, 8000, 2),
        TopoSpecTask('1080ti', 8<<30, 8000, 2),
    ], [[5000, 2180, 5000],
        [2180, 5000, 5000],
        [5000, 5000, 5000]])

    gdef = load('raw_data/{}/model.pickle'.format(m))
    prof_data = ProfileData(m)
    tge.simplify_graph(gdef, sinks=["Adam"])

    print(search(gdef, topo, prof_data))
