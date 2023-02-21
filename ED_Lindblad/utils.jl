export eigen_sparse, set_parameters


function eigen_sparse(x)
    decomp, history = partialschur(x, nev=1, which=LR(); restarts=100000); # only solve for the ground state
    vals, vecs = partialeigen(decomp);
    return vals, vecs
end

⊗(x,y) = kron(x,y)

id = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
sx = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
sy = [0.0+0.0im 0.0-1im; 0.0+1im 0.0+0.0im]
sz = [1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im]
sp = (sx+1im*sy)/2
sm = (sx-1im*sy)/2

sp_id = sparse(id)
sp_sx = sparse(sx)
sp_sy = sparse(sy)
sp_sz = sparse(sz)
sp_sp = sparse(sp)
sp_sm = sparse(sm)

mutable struct parameters
    N::Int
    dim::Int
    J::Float64
    h::Float64
    γ::Float64
    α::Float64
end

function set_parameters(N,J,h,γ,α)
	params.N = N;
    params.dim = 2^N;
    params.J = J;
    params.h = h;
    params.γ = γ;
    params.α = α;
end

function calculate_Kac_norm(params::parameters; offset=0.0) #periodic BCs only!
    N_K = offset
    for i in 1:convert(Int64,floor(params.N/2))
        N_K+=1/i^params.α
    end
    return N_K
end