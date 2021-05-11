import numpy as np
import tensorflow as tf
from utils import save, load, info
from baseline import gen_baselines, eval_baselines, random_cross_node, random_shuffle_node, Trace
from environment import evaluate_with_feedback

records = load("records")

record = records[8]
baseline = record['baselines'][1]
evaluate_with_feedback(record, baseline.nodemask, baseline.ncclmask, baseline.psmask, "{}.json".format(baseline.name))

# for record in records:
#     if 'baselines' not in record:
#         baselines = gen_baselines(record)
#         eval_baselines(record, baselines)
#         record['baselines'] = baselines
#         save(records, "records")

# for record_id, record in enumerate(records):
#     info(record_id, [("*" if b.invalidity > 0 else "") + str(b.score) for b in record['baselines']])

# def trace(record_id):
#     record = records[record_id]
#     for baseline_id in range(4):
#         baseline = record['baselines'][baseline_id]
#         evaluate_with_feedback(record, baseline.nodemask, baseline.ncclmask, baseline.psmask, "{}.json".format(baseline.name))
