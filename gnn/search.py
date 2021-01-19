import tensorflow as tf
import numpy as np
from environment import fitness
from utils import save, load, info

from pymoo.model.problem import Problem
from pymoo.algorithms.so_brkga import BRKGA
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize

from multiprocessing import Pool

pool = Pool(16)

class MyProblem(Problem):
    def __init__(self, record):
        self.record = record
        n = 3 * len(record['op_groups']) * record['topo_spec'].ntasks + len(record['op_groups']) * (1 + record['topo_spec'].ntasks)
        super().__init__(n_var=n, n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        n, m = len(self.record['op_groups']), self.record['topo_spec'].ntasks
        phenos = []
        for i in range(x.shape[0]):
            placement = np.argmax(np.reshape(x[i, :n*m*3], (n*m, 3)), axis=1)
            communication = np.argmax(np.reshape(x[i, n*m*3:], (n, m+1)), axis=1)
            phenos.append(np.hstack([placement, communication]))

        ks = pool.map(fitness, [(self.record, pheno) for pheno in phenos])

        out["F"] = [[k] for k in ks]
        out["pheno"] = np.array(phenos)

class MySampling(Sampling):
    def __init__(self, placement):
        super().__init__()
        self.p = placement

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.rand(n_samples, problem.n_var)

        n, m = self.p.shape
        for s in range(n_samples):
            for p in range(n*m*3):
                i, j, k = np.unravel_index(p, (n, m, 3))
                if self.p[i, j] == 1 and k == 0:
                    X[s, p] **= 2
                elif self.p[i, j] == 0 and k != 0:
                    X[s, p] **= 2
            X[s, n*m*3:] **= 2

        return X

def search(record, placement, n_gen=15):
    problem = MyProblem(record)

    algorithm = BRKGA(
        n_elites=16,
        n_offsprings=32,
        n_mutants=16,
        bias=0.8,
        sampling=MySampling(placement),
        eliminate_duplicates=True)

    res = minimize(problem, algorithm, ("n_gen", n_gen), verbose=False)

    # info("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    return (res.F[0], *decode(record, res.opt.get("pheno")[0]))

def decode(record, pheno):
    nodemask = np.reshape(pheno[:len(record['op_groups']) * record['topo_spec'].ntasks], (len(record['op_groups']), record['topo_spec'].ntasks))
    ncclmask = (pheno[len(record['op_groups']) * record['topo_spec'].ntasks:] == 0).astype(int)
    psmask = pheno[len(record['op_groups']) * record['topo_spec'].ntasks:] - 1

    return nodemask, ncclmask, psmask
