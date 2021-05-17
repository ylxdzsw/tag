import numpy as np
from dataclasses import dataclass, field
from utils import groupby, car, cadr, cdr, info

def group_with_layer_name(gdef, base_groups, level=2, merge_gradient=True):
    group_table = {}
    for i, node in enumerate(gdef.node):
        name = node.name
        if merge_gradient:
            name = name.lstrip('gradients/')
            name = name.lstrip('Adam/update_')

        prefix = '/'.join(name.split('/')[:level])
        if prefix in group_table:
            group_table[prefix].append(i)
        else:
            group_table[prefix] = [i]

    groups = list(group_table.values())
    return compose(groups, base_groups)

def group_with_layer_name_tree(gdef, base_groups, limit=6, merge_gradient=True):
    @dataclass
    class Trie:
        name: str
        children: list = field(default_factory=list)
        nodes: list = field(default_factory=list)

        def append_node(self, name_seg, node_id):
            self.nodes.append(node_id)

            if name_seg.len() == 0:
                return

            head, *tail = name_seg
            for child in self.children:
                if child.name == head:
                    child.append_node(tail, node_id)

            child = Trie(head)
            self.children.append(child)
            child.append_node(tail, node_id)

        def dump_groups(self, groups):
            if self.nodes.len() < limit:
                groups.append(self.nodes)
            else:
                for child in self.children:
                    child.dump_groups(groups)
            return groups

    root = Trie('')
    for i, node in enumerate(gdef.node):
        name = node.name
        if merge_gradient:
            name = name.lstrip('gradients/')
            name = name.lstrip('Adam/update_')

        root.append_node(name.split('/'), i)

    return compose(root.dump_groups([]), base_groups)

def group_with_k_spanning_tree(gdef, base_groups, tensor_sizes, sinks, n_groups=20):
    pass

# TODO: use average time in all gpu types? weighted average?
def group_with_topk_nodes(gdef, base_groups, prof_data, n_groups=20):
    from utils import group_around_topk_costs

    cache = prof_data.get('1080ti', prof_data.maximum_batchsize())
    return group_around_topk_costs(gdef, base_groups, lambda x: cache[x], n_groups-1)

def group_with_topk_layers(gdef, base_groups, prof_data, n_groups=20):
    pass

def group_with_metis(gdef, base_groups, costs, batchsize, n_groups=20):
    from metis import metis
    id_list = metis(gdef, base_groups, costs, n_groups, list(range(len(gdef.node))), batchsize, balance_factor=5.)
    return list(groupby(enumerate(id_list), key=cadr, value=car).values())

def group_with_tge_basegroups(gdef):
    from tge import TGE

    base_groups = TGE(gdef, []).get_groups()
    return list(groupby(enumerate(base_groups), key=cadr, value=car).values())

# intersection of two grouping. Two nodes are in the same group if they are in the same group in either grouping scheme.
def compose(a, b):
    pass

# prim's algorithm
# alternative: https://networkx.github.io/documentation/stable/reference/algorithms/tree.html#module-networkx.algorithms.tree.mst
def k_spanning_tree(g, weights, k, seed=0):
    def get_weight(center, neighbor):
        return weights[ng.adj[center][neighbor][0]['id']]

    ng = g.to_networkx()
    tree_nodes = [seed]
    tree_edges = []
    while True:
        bridges = [(center, neighbor) for center in tree_nodes for neighbor in ng.adj[center] if neighbor not in tree_nodes ]
        if len(bridges) == 0:
            break
        highest_weight = np.max([ get_weight(center, neighbor) for center, neighbor in bridges ])
        index_of_edge_to_add = np.random.choice([ i for i, (center, neighbor) in enumerate(bridges) if get_weight(center, neighbor) == highest_weight ])
        center, neighbor = bridges[index_of_edge_to_add]
        tree_nodes.append(neighbor)
        tree_edges.append((center, neighbor, highest_weight))
    tree_edges.sort(key=lambda x: x[2])
    tree_edges = set( (center, neighbor) for center, neighbor, weight in tree_edges[k-1:] )
    groups = []
    for node in tree_nodes:
        for group in groups:
            for neighbor in group:
                if (node, neighbor) in tree_edges or (neighbor, node) in tree_edges:
                    group.append(node)
                    break
            else:
                continue
            break
        else:
            groups.append([node])

    return groups
