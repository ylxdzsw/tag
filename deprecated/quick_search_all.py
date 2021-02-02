import tensorflow as tf
import numpy as np
from environment import quick_fitness
from utils import save, load, info

from pymoo.model.problem import Problem
from pymoo.algorithms.so_brkga import BRKGA
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize

from multiprocessing import Pool

import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample, evaluate, base_strategies, score
from utils import save, load, info
from tge import TGE


pool = Pool(16)

class MyProblem(Problem):
    def __init__(self, record):
        self.record = record
        n = 3 * len(record['op_groups']) * record['topo_spec'].ntasks
        super().__init__(n_var=n, n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        n, m = len(self.record['op_groups']), self.record['topo_spec'].ntasks
        phenos = [np.argmax(np.reshape(x[i, :], (n*m, 3)), axis=1) for i in range(x.shape[0])]
        ks = pool.map(quick_fitness, [(self.record, pheno) for pheno in phenos])

        out["F"] = [[k] for k in ks]
        out["pheno"] = np.array(phenos)

class MySampling(Sampling):
    def __init__(self):
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        return np.random.rand(n_samples, problem.n_var)

def search(record, n_gen=50):
    problem = MyProblem(record)

    algorithm = BRKGA(
        n_elites=32,
        n_offsprings=32,
        n_mutants=32,
        bias=0.7,
        sampling=MySampling(),
        eliminate_duplicates=True)

    res = minimize(problem, algorithm, ("n_gen", n_gen), verbose=False)

    # info("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    return (res.F[0], decode(record, res.opt.get("pheno")[0]))

def decode(record, pheno):
    nodemask = np.reshape(pheno, (len(record['op_groups']), record['topo_spec'].ntasks))
    return nodemask

try:
    records = load("records")
    info("load saved records")
except:
    records = get_all_data()
    info("no saved records")
    save(records, "records")

for record_id, record in enumerate(records):
    if 'reference' not in record:
        record['reference'] = []
        for nodemask, ncclmask, psmask in base_strategies(record):
            loss_env = score(*evaluate(record, nodemask, ncclmask, psmask))
            record['reference'].append((loss_env, nodemask, ncclmask, psmask))
        save(records, "records")
    info([x[0] for x in record['reference']])

for record_id, record in enumerate(records):
    if (record_id % 8) not in (0, 2, 4):
        continue
    if 'reference' not in record:
        record['reference'] = []
        for nodemask, ncclmask, psmask in base_strategies(record):
            loss_env = score(*evaluate(record, nodemask, ncclmask, psmask))
            record['reference'].append((loss_env, nodemask, ncclmask, psmask))
        save(records, "records")

    info(search(record)[0], [x[0] for x in record['reference']])
