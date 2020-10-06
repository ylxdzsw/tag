import tensorflow as tf
import numpy as np
from environment import sample, evaluate, sample_and_evaluate, f
from utils import save, load, info

from pymoo.model.problem import Problem

from multiprocessing import Pool

pool = Pool(16)

class MyProblem(Problem):
    def __init__(self, record):
        self.record = record
        n = len(record['cgroups']) * len(record['devices'])
        super().__init__(n_var=n, n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        # pheno = (2.9999 * x).astype(int)
        pheno = x.astype(int)

        ks = pool.map(f, [(self.record, pheno[i, :]) for i in range(pheno.shape[0])])

        out["F"] = [[k] for k in ks]
        out["pheno"] = pheno
        # out["hash"] = hash(str(pheno))

from pymoo.algorithms.so_brkga import BRKGA
from pymoo.optimize import minimize

# from pymoo.model.duplicate import ElementwiseDuplicateElimination

# class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):
#     def is_equal(self, a, b):
#         info(a.get("hash"))
#         return a.get("hash") == b.get("hash")

from pymoo.model.sampling import Sampling

class MySampling(Sampling):
    def __init__(self, p):
        super().__init__()
        self.p = np.reshape(p, (-1,))

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), None, dtype=np.float)

        for i in range(n_samples):
            X[i, :] = np.random.rand(problem.n_var) < self.p

        # X = np.random.rand(n_samples, problem.n_var)

        return X

def search(record, p):
    problem = MyProblem(record)

    algorithm = BRKGA(
        n_elites=16,
        n_offsprings=48,
        n_mutants=16,
        bias=0.7,
        sampling=MySampling(p),
        eliminate_duplicates=True)
        # eliminate_duplicates=MyElementwiseDuplicateElimination)

    res = minimize(problem, algorithm, ("n_gen", 20), seed=1, verbose=False)
    sol = np.reshape(res.opt.get("pheno")[0], (len(record['cgroups']), len(record['devices'])))

    # info("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    # info("Solution", sol)

    return res.F[0], sol
