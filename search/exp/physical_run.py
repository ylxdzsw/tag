import time
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
import tge
import numpy as np
import itertools
import copy
from utils import info, load

import os
os.environ["TF_CONFIG"] = '{ "cluster": { "worker": ["10.28.1.24:3806", "10.28.1.25:3901", "10.28.1.26:3901"] }, "task": {"type": "worker", "index": 0} }'

def setup_workers(workers, protocol="grpc"):
    import urllib.request
    import time

    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)

setup_workers(["10.28.1.24:3806", "10.28.1.25:3901", "10.28.1.26:3901"])

devices = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1",
    "/job:worker/replica:0/task:2/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:1",
    "/job:worker/replica:0/task:2/device:GPU:2",
    "/job:worker/replica:0/task:2/device:GPU:3",
)

resolver = TFConfigClusterResolver()
cluster = resolver.cluster_spec()
dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
config = dist.update_config_proto(tf.ConfigProto())
config.ClearField("device_filters")
config.allow_soft_placement = True  # log_device_placement=True)
config.gpu_options.allow_growth = True
server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc", config=config)

# from profiler import NcclProfiler
# nccl_model = NcclProfiler(devices, server.target).profile()
# print(nccl_model)

# raise SystemExit

# {'/job:worker/replica:0/task:0/device:GPU:0': [0.9584070443184248, 217.88718452314646, 0.21059811925140928, 223.5791934731678], '/job:worker/replica:0/task:1/device:GPU:0': [0.3714652630154233, 267.32093923984735, 0.2108443866768448, 211.22138885962116], '/job:worker/replica:0/task:2/device:GPU:0': [0.5378632532562655, 452.3317530440628, 0.1939005431393334, 517.9556800667694], '/job:worker/replica:0/task:0/device:GPU:0,/job:worker/replica:0/task:1/device:GPU:0': [0.5212558559955102, 357.4631960568025, 0.2744714020116987, 305.07792803331535], '/job:worker/replica:0/task:0/device:GPU:0,/job:worker/replica:0/task:2/device:GPU:0': [-0.0374282614817747, 508.69988218682647, 0.2612657083390182, 286.2716917866893], '/job:worker/replica:0/task:1/device:GPU:0,/job:worker/replica:0/task:2/device:GPU:0': [0.11458482532105904, 492.3139957929465, 0.261508923354283, 293.04110118062215], '/job:worker/replica:0/task:0/device:GPU:0,/job:worker/replica:0/task:1/device:GPU:0,/job:worker/replica:0/task:2/device:GPU:0': [0.1472075204785818, 488.6477998063673, 0.3076460973921253, 0.00013946418923570407]}

import sys
data_path = sys.argv[1]

gdef, prof_data, batchsize, strategy = load(data_path)

g = (tge.TGE(gdef, devices)
    .set_strategy(strategy)
    .replace_placeholder(batchsize)
    # .verbose()
    .compile()
    .get_result()
)

tf.import_graph_def(g)
graph = tf.get_default_graph()

opt = graph.get_operation_by_name("import/Adam/replica_0")
init = graph.get_operation_by_name("import/init/replica_0")

sess = tf.Session(server.target, config=config)
sess.run(init)
sess.run(opt)

run_meta = tf.compat.v1.RunMetadata()
run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
sess.run(opt, options=run_opt, run_metadata=run_meta)

with open("meta_{}.pb".format(data_path), "w") as fo:
    fo.write(pbtf.MessageToString(run_meta))

tl = timeline.Timeline(run_meta.step_stats)
with open("timeline_{}.json".format(data_path), "w") as fo:
    fo.write(tl.generate_chrome_trace_format())

for _ in range(3):
    sess.run(opt)
tic = time.perf_counter()
for _ in range(10):
    sess.run(opt)
toc = time.perf_counter()

print("time: {}".format((toc - tic) / 10))
