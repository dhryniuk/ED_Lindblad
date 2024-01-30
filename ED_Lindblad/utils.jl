export eigen_sparse, set_parameters

export id, sx, sy, sz, sp, sm


function eigen_sparse(x)
    decomp, history = partialschur(x, nev=1, which=LR(); restarts=100000); # only solve for the ground state
    vals, vecs = partialeigen(decomp);
    return vals, vecs
end

export eigen_sparse2
function eigen_sparse2(x)
    decomp, history = partialschur(x, nev=2, which=LR(); restarts=100000); # only solve for the ground state
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
    N::Int64
    dim::Int64
    J::Float64
    h::Float64
    hz::Float64
    γ_l::Float64
    γ_d::Float64
    α::Float64
end

function set_parameters(N,J,h,hz,γ_l,γ_d,α)
	params.N = N;
    params.dim = 2^N;
    params.J = J;
    params.h = h;
    params.hz = hz;
    params.γ_l = γ_l;
    params.γ_d = γ_d;
    params.α = α;
end

function HarmonicNumber(n::Int,α::Float64)
    h=0
    for i in 1:n
        h+=i^(-α)
    end
    return h
end

function calculate_Kac_norm(params::parameters)
    N = params.N
    α = params.α

    if mod(N,2)==0
        return (2*HarmonicNumber(1+N÷2,α) - 1 - (1+N÷2)^(-α))
    else
        return (2*HarmonicNumber(1+(N-1)÷2,α) - 1)
    end
end

function old_calculate_Kac_norm(params::parameters; offset=0.0) #periodic BCs only!
    N_K = offset
    for i in 1:convert(Int64,floor(params.N/2))
        N_K+=1/i^params.α
    end
    return N_K
end