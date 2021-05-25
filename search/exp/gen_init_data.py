from dataclasses import dataclass
from typing import Any
import numpy as np
import time
import collections

from train import playout
from model import Model, policy
from environment import evaluate_with_feedback, invalidity
from utils import save, load, info
from mcts import Tree, Action
from data import ProfileData, estimate_model_size, gen_random_topology, gen_data

import sys
import tge

m = np.random.choice(["inception", "resnet", "vgg", "transformer", "bert", "berts"])
gdef = load('raw_data/{}/model.pickle'.format(m))
prof_data = ProfileData(m)
tge.simplify_graph(gdef, sinks=["Adam"])

model_size = estimate_model_size(gdef, prof_data.maximum_batchsize())
topo = gen_random_topology(model_size)
record = gen_data(gdef, prof_data, prof_data.maximum_batchsize(), topo)
record['model_name'] = m

i = int(sys.argv[1])

d = {
    "inception": 4,
    "resnet": 4,
    "vgg": 8,
    "transformer": 1,
    "bert": 1,
    "berts": 4
}

data = []
for _ in range(d[m]):
    data.extend(playout(record))

save(data, f"init_data/{i}")
