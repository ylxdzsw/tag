import time
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from utils import load

devices = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1"
)

gdef = load("raw_data/rnnlm2x/model.pickle")
prof_dict = load("raw_data/rnnlm2x/1080ti/60.pickle")

import tge
tge.simplify_graph(gdef, sinks=['Adam'])

# options = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 1]]
# strategy = { node.name: [np.random.randint(0, 2)] + options[np.random.randint(0, len(options))] for node in gdef.node }

for node in gdef.node:
    if len(node.input) > 16:
        node.input[:] = [node.input[0]]

def f(x):
    if x.startswith("gradients/lstm/") or x.startswith("lstm/") or x.startswith("Adam/update_lstm/"):
        return [1, 1, 1, 0, 0]
    elif x.startswith("gradients/lstm_1/") or x.startswith("lstm_1/") or x.startswith("Adam/update_lstm_1/"):
        return [1, 0, 0, 1, 1]
    else:
        return [1, 1, 1, 1, 1]

strategy = { node.name: f(node.name) for node in gdef.node }

prof_dict = { (name, nrep): [ prof_dict[name] // nrep ] * 4 for name, value in prof_dict.items() for nrep in [1, 2, 3, 4] }


g = (tge.TGE(gdef, devices, sinks=['Adam'])
    .custom(strategy)
    .fill_batchsize(60)
    .replace_placeholder(60)
    # .use_collective()
    .set_bandwidth(intra=28100, inter=6000)
    .heft(prof_dict, True)
    .evaluate(prof_dict, "simulated.json")
)

print(g)
