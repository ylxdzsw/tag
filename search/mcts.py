import numpy as np
import itertools
import copy
import tge
from dataclasses import dataclass
from typing import Any
from data import TopoSpec, TopoSpecTask, ProfileData, device_name, gen_topology_for_simulator, gen_nccl_model, gen_data
from grouping import group_with_topk_nodes, group_with_tge_basegroups
from utils import info, load
from metis import metis
from environment import evaluate_with_feedback, invalidity

@dataclass
class Action:
    placement: Any # a list of the same length of machines
    communication: Any # 0: PS, 1: NCCL, 2: MP

    def to_mask(self):
        placement_mask = np.expand_dims(np.array(self.placement_mask), 1)
        communication_mask = np.zeros((1, 3))
        communication_mask[0, self.communication] = 1
        return placement_mask, communication_mask

@dataclass
class State:
    record: Any
    baseline: Any # (time, action)
    actions: Any # the actions taken so far. The rest nodes uses the first action (same strategy as the most computational expensive group)
    feedback: Any

    # shallow copy except for the actions
    def clone(self):
        x = copy.copy(self)
        x.actions = copy.deepcopy(self.actions)
        return x

    def finished(self):
        return len(self.actions) >= len(self.record['op_groups'])

    def fill_cache(self):
        gdef, topo_spec, prof_data, batchsize = self.record['gdef'], self.record['topo_spec'], self.record['prof_data'], self.record['batchsize']

        state_copy = self.clone()
        base_action = ([1 for _ in range(len(topo_spec.tasks))], 1)
        state_copy.actions = [base_action]
        time, feedback = evaluate_with_feedback(state_copy)

        if invalidity(state_copy.record, feedback) > 0: # if OOM, use MP as baseline
            base_action = ([1 for _ in range(len(topo_spec.tasks))], 2)
            state_copy.actions = [base_action]
            time, feedback = evaluate_with_feedback(state_copy)

        # TODO: save cache to record? or generate this in data.py?
        self.baseline = time, base_action
        return self

    def get_action(self, i):
        if len(self.actions) == 0:
            return self.baseline[1]
        if i < len(self.actions):
            return self.actions[i]
        else:
            return self.actions[0]

    def dump_strategy(self):
        record = self.record
        gdef = record['gdef']
        devices = record['device_list']
        batchsize = record['batchsize']

        strategy = {}
        for gid, group in enumerate(self.record['op_groups']):
            action = self.get_action(gid)

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
                costs = [ int(self.record['parameter_sizes'][i] / 100000) for i in group ]
                # single card placement does not have base_group restriction.
                assignments = metis(gdef, [[i] for i in range(len(gdef.node))], costs, len(placed_devices), group, batchsize, balance_factor=1.5)
                for node_id, assignment in zip(group, assignments):
                    s = [0] * (1 + len(devices))
                    s[placed_devices[assignment]+1] = 1
                    strategy[gdef.node[node_id].name] = s

        return strategy

    @staticmethod
    def new(record):
        return State(record, 0, [], None).fill_cache()

class Node:
    def __init__(self, action):
        self.action = action
        self.p = 0
        self.q = 0
        self.c = 0
        self.n_visits = 0
        self.children = []
        self.value = None

    def playout_and_update_recursive(self, state, options):
        if self.is_leaf():
            if not state.finished():
                self.expand(state, options)
            if len(state.actions) == 0: # root at first
                return 0.
            leaf_value = self.evaluate(state)
            self.update(leaf_value)

            return leaf_value
        child = self.select_child()
        state.actions.append(child.action)
        leaf_value = child.playout_and_update_recursive(state, options)
        self.update(leaf_value)
        return leaf_value

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        return max(self.children, key=lambda x: x.puct(self.n_visits))

    def puct(self, pvisit):
        return self.q + self.c * self.p * np.sqrt(pvisit) / (1 + self.n_visits)

    def update(self, leaf_value):
        self.n_visits += 1
        self.q += (leaf_value - self.q) / self.n_visits

    def expand(self, state, options):
        for placement in itertools.product([0, 1], repeat=len(state.record['topo_spec'].tasks)):
            if sum(placement) == 0:
                continue

            ndevices = sum( state.record['topo_spec'].tasks[i].number for i in placement if i == 1 )

            for communication in range(3):
                if ndevices == 1 and communication != 0:
                    continue

                if options.real_topo and state.record['batchsize'] % ndevices != 0 and communication != 2:
                    continue

                action = placement, communication
                self.children.append(Node(action))

        for child in self.children:
            child.c = np.sqrt(len(self.children) / 10)

        if options.policy_fun is not None:
            masks = [ child.action.to_mask() for child in self.children ]
            log_softmaxs = policy_fun(state, *zip(*masks))

            for child, log_softmax in zip(self.children, log_softmaxs):
                info(child.action, np.exp(log_softmax))
                child.p = np.exp(log_softmax)
        else:
            for child in self.children:
                child.p = 1 / len(self.children)

        np.random.shuffle(self.children)

    def evaluate(self, state):
        if self.value is None:
            time, feedback = evaluate_with_feedback(state)

            # if invalidity(state.record, feedback) > 0:
            #     info("OOM")
            # info(state.baseline[0], time, [x / 1000_000 for x in feedback["peak_memory"]])

            speed_up = -1 if invalidity(state.record, feedback) > 0 else state.baseline[0] / time - 1
            if speed_up > 1:
                speed_up = np.sqrt(np.sqrt(speed_up))
            self.value = speed_up
            state.feedback = feedback
        return self.value

class Tree:
    def __init__(self, policy_fun, real_topo=False): # real_topo controls whether we should filter out the un-dividable replications
        self.policy_fun = policy_fun
        self.real_topo = real_topo
        self.root = Node(None)

    def playout(self, state, ntimes, trace_fun=None):
        best = -1
        best_actions = None
        for n in range(ntimes):
            state_clone = state.clone()
            leaf_value = self.root.playout_and_update_recursive(state_clone, self)
            if leaf_value > best:
                best = leaf_value
                best_actions = state_clone.actions
            if trace_fun is not None:
                trace_fun(leaf_value, state_clone.actions)
        return best, best_actions

    def get_action(self):
        return max(self.root.children, key=lambda x: x.n_visits).action

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

    record = gen_data(gdef, prof_data, prof_data.maximum_batchsize(), topo)

    state = State(record, None, 0, []).fill_cache()
    print(Tree(None).playout(state, 800))
