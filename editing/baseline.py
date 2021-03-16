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
    score: int = 0
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

def random_cross_node(record, *strategies):
    n = len(record['gdef'].node)
    m = len(record['devices'])

    result_nodemask = np.ones((n, m), dtype=np.int)
    result_ncclmask = [0] * n
    result_psmask = [0] * n

    for i in range(n):
        j = np.random.randint(len(strategies))
        nodemask, ncclmask, psmask = strategies[j]
        result_nodemask[i, :] = nodemask[i, :]
        result_ncclmask[i] = ncclmask[i]
        result_psmask[i] = psmask[i]

    return result_nodemask, result_ncclmask, result_psmask

def random_shuffle_node(record, nodemask, ncclmask, psmask):
    indexes = np.arange(len(record['gdef'].node))
    np.random.shuffle(indexes)
    return nodemask[indexes, :], np.array(ncclmask)[indexes], np.array(psmask)[indexes]

@dataclass
class Trace:
    nodemask: Any
    ncclmask: Any
    psmask: Any
    score: int = 0
    feedback: Any = None
    next: Any = None

    def evaluate_with_feedback(self, record):
        self.score, self.feedback = evaluate_with_feedback(record, self.nodemask, self.ncclmask, self.psmask)

    def similarity(self, trace):
        return (
            float(np.sum(self.nodemask == trace.nodemask)) +
            float(np.sum(np.array(self.ncclmask) == np.array(trace.ncclmask))) +
            float(0.5 * np.sum(np.array(self.psmask) == np.array(trace.psmask)))
        )
