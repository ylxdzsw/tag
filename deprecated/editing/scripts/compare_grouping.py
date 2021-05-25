import grouping
from utils import load
import numpy as np

records = load('records')

def parameter_sizes(record, grouping):
    psizes = record['op_feats'][:, 4] * record['scaler'][0] * record['scaler'][1]
    spsizes = [ sum(times[group]) for group in grouping ]
    return min(spsizes), max(spsizes), np.std(spsizes)

def cross_tensor_size(record, grouping):
    pass

def computation_times(record, grouping):
    times = record['op_feats'][:, 0]
    stimes = [ sum(times[group]) for group in grouping ]
    return min(stimes), max(stimes), np.std(stimes)

def dump_grouping(record, grouping, name):
    nodes = record['gdef'].node
    prof_data = record['prof_data'].get('1080ti', record['prof_data'].maximum_batchsize())
    with open('grouping_{}'.format(name), 'w') as f:
        for gid, group in enumerate(grouping):
            f.write('group ' + str(gid) + ':\n')
            for node in sorted(group):
                f.write(nodes[node].name + ' ' + nodes[node].op + ' ' + str(record['op_feats'][node, 0]) + ' ' + str(prof_data[nodes[node].name]) + '\n')

grouping = grouping.group_with_tge_basegroups(records[1]['gdef'])

print(computation_times(records[1], grouping))
print(parameter_sizes(records[1], grouping))
dump_grouping(records[1], grouping, 'base')
