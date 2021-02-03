import numpy as np
from dataclasses import dataclass
from typing import Any
from metis import metis
from environment import evaluate_with_feedback, invalidity

@dataclass
class BaseLine:
    name: str
    nodemask: Any = None
    ncclmask: Any = None
    psmask: Any = None
    score: Any = None
    invalidity: int = 0
    feedback: Any = None

def gen_baselines(record):
    baselines = []

    n = len(record['gdef'].node)
    m = len(record['devices'])

    x = BaseLine('gpu0')
    s = np.zeros((n, m), dtype=np.int)
    s[:, 0] = 1
    x.nodemask = s
    x.ncclmask = [0] * n
    x.psmask = [0] * n
    baselines.append(x)

    x = BaseLine('dp-nccl')
    x.nodemask = np.ones((n, m), dtype=np.int)
    x.ncclmask = [1] * n
    x.psmask = [0] * n
    baselines.append(x)

    x = BaseLine('dp-ps-round-robin')
    x.nodemask = np.ones((n, m), dtype=np.int)
    x.ncclmask = [0] * n
    x.psmask = [ i % m for i in range(n) ]
    baselines.append(x)

    x = BaseLine('mp-metis')
    s = np.zeros((n, m), dtype=np.int)
    _, assignments = metis(record["gdef"], record["prof_data"], m, range(n), record["batchsize"])
    for i, a in enumerate(assignments):
        s[i, a] = 1
    x.nodemask = s
    x.ncclmask = [0] * n
    x.psmask = [0] * n
    baselines.append(x)

    return baselines

def eval_baselines(record, baselines):
    for b in baselines:
        score, feedback = evaluate_with_feedback(record, b.nodemask, b.ncclmask, b.psmask)
        b.score = score
        b.invalidity = invalidity(record, feedback)
        b.feedback = feedback
