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

import sys
gdef, prof_data, batchsize, strategy = load(sys.argv[1])

N = len(strategy)
device_count = [0 for _ in list(strategy.values())[0][1:]]
communication_count = [0, 0]
for s in strategy.values():
    if s[0] < 0:
        communication_count[0] += 1
    elif s[0] == 1:
        communication_count[1] += 1
    for i in range(len(s) - 1):
        device_count[i] += s[i+1]

print(*[ c / N for c in communication_count ])
print(*[ c / N for c in device_count ])
