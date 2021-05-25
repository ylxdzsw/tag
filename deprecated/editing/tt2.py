import numpy as np
import tensorflow as tf
import tge
from utils import save, load, info
from baseline import gen_baselines, eval_baselines, random_cross_node, random_shuffle_node, Trace
from environment import evaluate_with_feedback
from data import ProfileData, gen_data, TopoSpec, TopoSpecTask

topo = TopoSpec([
    TopoSpecTask('1080ti', 6<<30, 50000, 2),
    TopoSpecTask('v100',   8<<30, 50000, 1),
], [[2810, 2810],
    [2810, 2810]])

gdef = load('raw_data/inception/model.pickle')
prof_data = ProfileData('inception')
tge.simplify_graph(gdef, sinks=["Adam"])

record = gen_data(gdef, prof_data,  prof_data.maximum_batchsize(), topo)
baselines = gen_baselines(record)
eval_baselines(record, baselines)
record['baselines'] = baselines

nodemask = np.array([
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [0, 0, 1],
       ])

evaluate_with_feedback(record, nodemask, baseline.ncclmask, baseline.psmask, "{}.json".format('test'))

evaluate_with_feedback(record, baseline.nodemask, baseline.ncclmask, baseline.psmask, "{}.json".format(baseline.name))
