include("ED_Lindblad/ED_Lindblad.jl")
using .ED_Lindblad
using LinearAlgebra
using SparseArrays

const N = 9
const dim = 2^N
const h = 1.0
const J = -0.5
const γ = 0.5
const α = 1.0

ED_Lindblad.set_parameters(N,J,h,γ,α)

#@code_warntype 
#L = ED_Lindblad.DQIM(ED_Lindblad.params, "periodic")
#@time L = ED_Lindblad.sparse_DQIM(ED_Lindblad.params, "periodic")
#@time L = ED_Lindblad.sparse_LRInt_DQIM(ED_Lindblad.params, "periodic")
#@time L = ED_Lindblad.sparse_LRInt_DQIM(ED_Lindblad.params, "periodic")
@time L = ED_Lindblad.sparse_LRDisp_DQIM(ED_Lindblad.params, "periodic")

display(L)

@time evals, evecs = eigen_sparse(L)

display(evals)