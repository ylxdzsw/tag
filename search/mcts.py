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
from scipy.special import softmax

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
    baseline: Any # (time, action) TODO: move to record
    actions: Any # the actions taken so far. The rest nodes uses the first action (same strategy as the most computational expensive group)
    result: Any # (speedup, feedback)

    def finished(self):
        return len(self.actions) >= len(self.record['op_groups'])

    def init(self):
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
        self.result = (-1 if invalidity(state_copy.record, feedback) > 0 else 0), feedback
        return self

    def evaluate(self):
        if self.result is None:
            time, feedback = evaluate_with_feedback(self)

            speed_up = -1 if invalidity(self.record, feedback) > 0 else self.baseline[0] / time - 1
            if speed_up > 1:
                speed_up = np.sqrt(np.sqrt(speed_up))
            self.result = speed_up, feedback
        return self.result

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
        return State(record, None, [], None).init()

    # shallow copy except for the actions, and optionally append an action to the copy
    def clone(self, action=None):
        new_actions = [ action for action in self.actions ]
        if action is not None:
            new_actions.append(action)

        return State(self.record, self.baseline, new_actions, None)

class Node:
    def __init__(self, state, action):
        self.action = action
        self.state = state # the state after taking the action
        self.p = 0 # prior probability of choosing this among all siblings
        self.q = 0 # the average speedup when taking this action on this state
        self.c = 0 # per-node parameter to control exploration vs exploitation
        self.n_visits = 0
        self.children = []

    def playout_and_update_recursive(self, options):
        if self.is_leaf():
            if not self.state.finished():
                self.expand(options)
            value, _ = self.state.evaluate()
            self.update(value)
            return self.state

        child = self.select_child()
        leaf_state = child.playout_and_update_recursive(options)
        self.update(leaf_state.result[0])
        return leaf_state

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        return max(self.children, key=lambda x: x.puct(self.n_visits))

    def puct(self, pvisit):
        return self.q + self.c * self.p * np.sqrt(pvisit) / (1 + self.n_visits)

    def update(self, leaf_value):
        self.n_visits += 1
        self.q += (leaf_value - self.q) / self.n_visits

    def expand(self, options):
        record = self.state.record

        for placement in itertools.product([0, 1], repeat=len(record['topo_spec'].tasks)):
            if sum(placement) == 0:
                continue

            ndevices = sum( record['topo_spec'].tasks[i].number for i in placement if i == 1 )

            for communication in range(3):
                if ndevices == 1 and communication != 0:
                    continue

                if options.real_topo and record['batchsize'] % ndevices != 0 and communication != 2:
                    continue

                action = placement, communication
                new_state = self.state.clone(self.action)
                self.children.append(Node(new_state, action))

        for child in self.children:
            child.c = np.sqrt(len(self.children) / 10)

        if options.policy_fun is not None:
            log_ps = policy_fun(self.state, (child.action for child in self.children))

            for child, log_p in zip(self.children, log_ps):
                info(child.action, np.exp(log_p))
                child.p = np.exp(log_p)
        else:
            for child in self.children:
                child.p = 1 / len(self.children)

        np.random.shuffle(self.children)

class Tree:
    def __init__(self, record, policy_fun, real_topo=False): # real_topo controls whether we should filter out the un-dividable replications
        self.policy_fun = policy_fun
        self.real_topo = real_topo
        self.root = Node(State.new(record), None)

    def playout(self, ntimes, trace_fun=None):
        best = None
        for n in range(ntimes):
            leaf_state = self.root.playout_and_update_recursive(self)
            if best == None or leaf_state.result[0] > best.result[0]:
                best = leaf_state
            if trace_fun is not None:
                trace_fun(leaf_state.result[0], leaf_state.actions)
        return best

    def get_actions_and_probs(self):
        assert len(self.root.children) > 0
        visits = [ np.log(child.n_visits + 1e-10) for child in self.root.children ]
        return [ child.action for child in self.root.children ], softmax(visits)

    def chroot(self, i):
        self.root = self.root.children[i]

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
    best = Tree(record, None).playout(800)
    print(best.result[0], best.actions)
