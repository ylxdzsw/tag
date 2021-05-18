from grouping import group_with_tge_basegroups, group_with_topk_nodes
from utils import load
import numpy as np

record = load('records')[-1]

# def parameter_sizes(record, grouping):
#     psizes = record['op_feats'][:, 4] * record['scaler'][0] * record['scaler'][1]
#     spsizes = [ sum(times[group]) for group in grouping ]
#     return min(spsizes), max(spsizes), np.std(spsizes)

# def cross_tensor_size(record, grouping):
#     pass

# def computation_times(record, grouping):
#     times = record['op_feats'][:, 0]
#     stimes = [ sum(times[group]) for group in grouping ]
#     return min(stimes), max(stimes), np.std(stimes)

def dump_grouping(record, grouping, name):
    nodes = record['gdef'].node
    prof_data = record['prof_data'].get('1080ti', record['prof_data'].maximum_batchsize())
    with open('grouping_{}'.format(name), 'w') as f:
        for gid, group in enumerate(grouping):
            f.write('group ' + str(gid) + ':\n')
            for node in sorted(group):
                f.write(nodes[node].name + ' ' + nodes[node].op + ' ' + str(prof_data[nodes[node].name]) + '\n')

grouping = record['op_groups']
dump_grouping(record, grouping, 'metis')

grouping = group_with_tge_basegroups(record['gdef'])
dump_grouping(record, grouping, 'base')

grouping = group_with_topk_nodes(record['gdef'], grouping, record['prof_data'], 40)
dump_grouping(record, grouping, 'topk')
