import time
import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from utils import info, load

devices = (
    "GPU:0",
    "GPU:1",
)

gdef, _, _, _ = load("bert_1080ti.pickle")

import tge

x = [21422, 21425, 21437, 21453]
x = [] # [33, 35, 39, 48, 60, 62, 69, 71, 72, 91, 93, 96, 98, 99, 103, 105, 109, 111, 113, 121, 122, 123, 124, 125, 130, 132, 133, 138, 139, 140, 141, 166, 6897, 6902, 6906, 6909, 6912, 7014, 7017, 7185, 7190, 7194, 7197, 7200, 7273, 7276, 7326, 7329, 7331, 7334, 7502, 7507, 7511, 7514, 7517, 7619, 7622, 7790, 7795, 7799, 7802, 7805, 7878, 7881, 7931, 7934, 7936, 7939, 8107, 8112, 8116, 8119, 8122, 8224, 8227, 8395, 8400, 8404, 8407, 8410, 8483, 8486, 8536, 8539, 8541, 8544, 8712, 8717, 8721, 8724, 8727, 8829, 8832, 9000, 9005, 9009, 9012, 9015, 9088, 9091, 9141, 9144, 9146, 9149, 9317, 9322, 9326, 9329, 9332, 9434, 9437, 9605, 9610, 9614, 9617, 9620, 9693, 9696, 9746, 9749, 9751, 9754, 9922, 9927, 9931, 9934, 9937, 10039, 10042, 10210, 10215, 10219, 10222, 10225, 10298, 10301, 10351, 10354, 10356, 10359, 10527, 10532, 10536, 10539, 10542, 10644, 10647, 10815, 10820, 10824, 10827, 10830, 10903, 10906, 10956, 10959, 10961, 10964, 11132, 11137, 11141, 11144, 11147, 11249, 11252, 11420, 11425, 11429, 11432, 11435, 11508, 11511, 11561, 11564, 11566, 11569, 11737, 11742, 11746, 11749, 11752, 11854, 11857, 12025, 12030, 12034, 12037, 12040, 12113, 12116, 12166, 12169, 12171, 12174, 12342, 12347, 12351, 12354, 12357, 12459, 12462, 12630, 12635, 12639, 12642, 12645, 12718, 12721, 12771, 12774, 12776, 12779, 12947, 12952, 12956, 12959, 12962, 13064, 13067, 13235, 13240, 13244, 13247, 13250, 13323, 13326, 13376, 13379, 13381, 13384, 13552, 13557, 13561, 13564, 13567, 13669, 13672, 13840, 13845, 13849, 13852, 13855, 13928, 13931, 13981, 13984, 13986, 13989, 14157, 14162, 14166, 14169, 14172, 14274, 14277, 14445, 14450, 14454, 14457, 14460, 14533, 14536, 14586, 14589, 14591, 14594, 14762, 14767, 14771, 14774, 14777, 14879, 14882, 15050, 15055, 15059, 15062, 15065, 15138, 15141, 15191, 15194, 15196, 15199, 15367, 15372, 15376, 15379, 15382, 15484, 15487, 15655, 15660, 15664, 15667, 15670, 15743, 15746, 15796, 15799, 15801, 15804, 15972, 15977, 15981, 15984, 15987, 16089, 16092, 16260, 16265, 16269, 16272, 16275, 16348, 16351, 16401, 16404, 16406, 16409, 16577, 16582, 16586, 16589, 16592, 16694, 16697, 16865, 16870, 16874, 16877, 16880, 16953, 16956, 17006, 17009, 17011, 17014, 17182, 17187, 17191, 17194, 17197, 17299, 17302, 17470, 17475, 17479, 17482, 17485, 17558, 17561, 17611, 17614, 17616, 17619, 17787, 17792, 17796, 17799, 17802, 17904, 17907, 18075, 18080, 18084, 18087, 18090, 18163, 18166, 18216, 18219, 18221, 18224, 18392, 18397, 18401, 18404, 18407, 18509, 18512, 18680, 18685, 18689, 18692, 18695, 18768, 18771, 18821, 18824, 18826, 18829, 18997, 19002, 19006, 19009, 19012, 19114, 19117, 19285, 19290, 19294, 19297, 19300, 19373, 19376, 19426, 19429, 19431, 19434, 19602, 19607, 19611, 19614, 19617, 19719, 19722, 19890, 19895, 19899, 19902, 19905, 19978, 19981, 20031, 20034, 20036, 20039, 20207, 20212, 20216, 20219, 20222, 20324, 20327, 20495, 20500, 20504, 20507, 20510, 20583, 20586, 20636, 20639, 20641, 20644, 20812, 20817, 20821, 20824, 20827, 20929, 20932, 21100, 21105, 21109, 21112, 21115, 21188, 21191, 21241, 21244, 21246, 21249, 21422, 21425, 21437, 21453]
g = (tge.TGE(gdef, devices)
    .set_strategy({ node.name: [4 if i in x else 1, 1, 1] for i, node in enumerate(gdef.node) })
    .replace_placeholder(1)
    .compile()
    .get_result()
)

tf.import_graph_def(g)
graph = tf.get_default_graph()
for op in graph.get_operations():
    if 'embeddings' in op.name:
        op._set_device('/CPU:0')
        continue

opt = graph.get_operation_by_name("import/Adam/replica_0")
init = graph.get_operation_by_name("import/init/replica_0")
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)

p = time.perf_counter()
for i in range(10):
    sess.run(opt)

q = time.perf_counter()
info("time:", (q - p) / 10)
