import numpy as np
from cylp.cy import CyCbcModel, CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray

def solve_sfb(record, strategy):
    pass

def solve(B, T, M, S, batchsize, grads, α=1):
    nt = len(T)
    ns = len(S) # S: (i, j) -> tensorsize[i, j]

    S = { (int(key.split(',')[0]), int(key.split(',')[1])) : value for key, value in S.items() }

    Skeys = list(S.keys())
    Svalues = [ S[key] for key in Skeys]

    λ = np.ones(nt)

    model = CyLPModel()

    x = model.addVariable('x', nt, isInt=True)
    b = model.addVariable('b', ns, isInt=True)

    model += 0 <= x <= 1
    model += 0 <= b <= 1

    model.objective = (
        (M - 1) * sum(λ[i] * T[i] * x[i] for i in range(nt)) +
        M * (M - 1) / B * sum(Svalues[i] * b[i] for i in range(ns)) -
        2 * (M - 1) / M / B * sum(S[(gi, gj)] * x[gi] for gi, gj in grads) +
        α * sum(x)
    )

    for i in range(nt):
        if i in [ gi for gi, gj in Skeys ]:
            continue
        model += x[i] <= sum(x[j] for k, j in Skeys if i == k)

    for k in range(ns):
        i, j = Skeys[k]
        model += b[k] >= x[j] - x[i]

    s = CyClpSimplex(model)
    cbcModel = s.getCbcModel()
    cbcModel.solve()

    sol_x = cbcModel.primalVariableSolution['x']

    for i, v in enumerate(sol_x):
        if abs(v - 1) <= 10**-6:
            solutions.append(i)

    return solutions
