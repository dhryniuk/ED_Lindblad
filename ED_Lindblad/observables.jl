export magnetization


function magnetization(op,ρ,params)
    first_term_ops = fill(id, params.N)
    first_term_ops[1] = op

    m::ComplexF64=0
    for _ in 1:params.N
        m += tr(ρ*foldl(⊗, first_term_ops))
        first_term_ops = circshift(first_term_ops,1)
    end

    return m/params.N
end