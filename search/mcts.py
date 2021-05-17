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

@dataclass
class State:
    record: Any
    sorted_groups: Any
    dp_time: Any
    actions: Any # the actions taken so far. The rest nodes uses the first action (same strategy as the most computational expensive group)

    # shallow copy except for the actions
    def clone(self):
        x = copy.copy(self)
        x.actions = copy.deepcopy(self.actions)
        return x

    def finished(self):
        return len(self.actions) >= len(self.sorted_groups)

    def fill_cache(self):
        gdef, topo_spec, prof_data = self.record['gdef'], self.record['topo_spec'], self.record['prof_data']
        batchsize = prof_data.maximum_batchsize()
        self.sorted_groups = sorted(self.record['op_groups'], key=lambda group: -np.sum([ prof_data.get('1080ti', batchsize)[gdef.node[node_id].name] for node_id in group ])) # largest computation time first

        state_copy = self.clone()
        state_copy.actions.append(([1 for _ in range(len(topo_spec.tasks))], 1))
        time, _ = evaluate_with_feedback(state_copy)

        # TODO: if OOM, use MP as baseline
        # TODO: save to record
        self.dp_time = time
        return self

    @staticmethod
    def new(record):
        state = State(record, None, 0, [])
        state.fill_cache()
        return state

class Node:
    def __init__(self, action):
        self.action = action
        self.p = 0
        self.q = 0
        self.n_visits = 0
        self.children = []
        self.value = None

    def playout_and_update_recursive(self, state, policy_fun):
        if self.is_leaf():
            if not state.finished():
                if policy_fun != None:
                    self.expand(state)
                else:
                    self.expand_uniform(state)
            if len(state.actions) == 0: # root at first
                return 0.
            leaf_value = self.evaluate(state)
            self.update(leaf_value)

            return leaf_value
        child = self.select_child()
        state.actions.append(child.action)
        leaf_value = child.playout_and_update_recursive(state, policy_fun)
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

    def expand(self, state, action_probs):
        pass

    def expand_uniform(self, state):
        for placement in itertools.product([0, 1], repeat=len(state.record['topo_spec'].tasks)):
            if sum(placement) == 0:
                continue

            for communication in range(3):
                if communication == 1 and sum(placement) == 1:
                    continue

                action = placement, communication
                child = Node(action)
                if len(state.actions) > 0 and action == state.actions[0]:
                    child.n_visits += self.n_visits

                self.children.append(child)

        for child in self.children:
            child.p = 1 / len(self.children)

    def evaluate(self, state):
        if self.value is None:
            time, feedback = evaluate_with_feedback(state)
            speed_up = -1 if invalidity(state.record, feedback) > 0 else state.dp_time / time - 1
            self.value = speed_up
        return self.value

class Tree:
    def __init__(self, policy_fun):
        self.policy_fun = policy_fun
        self.root = Node(None)

    def playout(self, state, ntimes, trace=None):
        best = -1
        best_actions = None
        for n in range(ntimes):
            state_clone = state.clone()
            leaf_value = self.root.playout_and_update_recursive(state_clone, self.policy_fun)
            if leaf_value > best:
                best = leaf_value
                best_actions = state_clone.actions
            if trace is not None:
                trace.append((leaf_value, state_clone.actions))
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
