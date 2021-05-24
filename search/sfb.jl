using JuMP
using Cbc
using JSON3
using LinearAlgebra
using SparseArrays

α = 1

data = JSON3.read(stdin);
B = data.B
T = data.T
M = data.M
batchsize = data.batchsize

grads = [ (x...,) .+ 1 for x in data.grads ]
sinks = [ gi for (gi, gj) in grads ]

S = Dict( ((parse.(Int, split(String(k), ',')) .+ 1)...,) => v for (k, v) in data.S )
Skeys = collect(keys(S))
Svalues = [ S[k] for k in Skeys ]

nt = length(T)
ns = length(S)

IM = sparse([ i for (i, j) in Skeys ], 1:ns, 1, nt, ns) .+ sparse([ j for (i, j) in Skeys ], 1:ns, -1, nt, ns) # the directed incidence matrix
# AM = sparse([ i for (i, j) in Skeys ], [ j for (i, j) in Skeys ], 1, nt, nt) # the directed adjacent matrix

λ = ones(nt)

model = Model(Cbc.Optimizer);

# @variable(model, 0 <= x[1:nt] <= 1);
# @variable(model, 0 <= b[1:ns] <= 1);

@variable(model, x[1:nt], Bin);
@variable(model, b[1:ns], Bin);

@objective(model, Min,
    (M - 1) * sum(λ[i] * T[i] * x[i] for i in 1:nt) +
    M * (M - 1) * sum(Svalues[i] * b[i] for i in 1:ns) / B -
    2(M - 1) / M * sum(S[(gi, gj)] * x[gi] for (gi, gj) in grads) / B +
    α * sum(x)
);

# K = I - sparse(sinks, sinks, 1, nt, nt) - AM;

# @constraint(model, K'x .<= 0);

@constraint(model, -b .- IM'x .<= 0);

optimize!(model)

JSON3.write(stdout, (
    result=findall(value.(x) .>= .9) .- 1,
    M, B, batchsize
))
