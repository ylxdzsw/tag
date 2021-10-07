import time
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
import tge
import numpy as np
import itertools
import copy
from utils import info, load, save

# g1: 10.28.1.16, p100 x 2
# g2: 10.28.1.17, p100 x 2
# g7: 10.28.1.22, 1080ti x 2
# g8: 10.28.1.23, 1080ti x 2
# g9: 10.28.1.24, 1080ti x 2
# g10: 10.28.1.25, 1080ti x 2
# g11: 10.28.1.26, v100 x 4
# g12: 10.28.1.27, v100+ x 4

from contextlib import contextmanager
@contextmanager
def measure_time(name):
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    print("{}: {:.3g}s".format(name, toc - tic))

import os
os.environ["TF_CONFIG"] = '{ "cluster": { "worker": ["10.28.1.24:3806", "10.28.1.16:3901"] }, "task": {"type": "worker", "index": 0} }'

def setup_workers(workers, protocol="grpc"):
    import urllib.request
    import time

    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)

setup_workers(["10.28.1.24:3806", "10.28.1.16:3901"])

devices = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1",
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

from profiler import NcclProfiler
nccl_model = NcclProfiler(devices, server.target).profile()
save(nccl_model, "g9-g1-nccl")
print(nccl_model)
raise SystemExit

import sys
data_path = sys.argv[1]

gdef, prof_data, batchsize, strategy = load(data_path)

strategy = { node.name: [1, 0, 0, 1, 1] for node in gdef.node }

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

with measure_time("init"):
    sess.run(init)
for i in range(10):
    with measure_time("warm up run round {}/10".format(i+1)):
        sess.run(opt)

print("start training")
tic = time.perf_counter()
for _ in range(50):
    sess.run(opt)
toc = time.perf_counter()

print("average time: {}".format((toc - tic) / 50))

gdef, prof_data, batchsize, _ = load(data_path)
nccl_model = load("g9-g1-nccl")

tge = tge.TGE(gdef, devices, sinks=["Adam"])
tge.set_strategy(strategy)
tge.fill_batchsize(batchsize)
tge.replace_placeholder(batchsize)
tge.set_bandwidth(inter=2810, intra=3000)
tge.set_nccl_model(nccl_model)

time, mem = tge.evaluate(prof_data)

print("simulated time: {}".format(time / 1000000))
