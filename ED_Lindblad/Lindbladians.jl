#export DQIM, sparse_DQIM, LRInt_DQIM, sparse_LRInt_DQIM, LRDisp_DQIM, sparse_LRDisp_DQIM, make_one_body_Lindbladian
export DQIM, sparse_DQIM, sparse_DQIM_local_dephasing, sparse_DQIM_collective_dephasing, sparse_DQIM_LRI
export sparse_DQIM_LRI_Schulz, sparse_DQIM_Schulz


function DQIM(params::parameters, boundary_conditions)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D
    """

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sx, boundary_conditions)
    H_Z = params.hz*one_body_Hamiltonian_term(params, sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X + H_Z)
    L_D = params.γ_l*one_body_Lindbladian_term(sm, params)

    return L_H + L_D
end

function sparse_DQIM(params::parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    H_Z = params.hz*one_body_Hamiltonian_term(params, sp_sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X + H_Z)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end

export sparse_DQIM_Cui

function sparse_DQIM_Cui(params::parameters, boundary_conditions)

    H_ZZ= params.J/4*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.h/2*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    H_Z = (params.hz-params.J/2)*one_body_Hamiltonian_term(params, sp_sz, boundary_conditions)
    H_boundries = params.J/4*(single_one_body_Hamiltonian_term(1, params, sp_sz, boundary_conditions) + single_one_body_Hamiltonian_term(params.N, params, sp_sz, boundary_conditions))
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X + H_Z + H_boundries)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end

function sparse_DQIM_Schulz(params::parameters, boundary_conditions)

    z=0.5*(sp_id+sp_sz)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, z, z, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    H_Z = params.hz*one_body_Hamiltonian_term(params, z, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X + H_Z)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end

function sparse_DQIM_local_dephasing(params::parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    #display(Matrix(L_H))
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)
    L_D_dephasing = params.γ_d*one_body_Lindbladian_term(sp_sz, params)
    #display(Matrix(L_D_dephasing))

    return L_H + L_D + L_D_dephasing
end

function sparse_DQIM_collective_dephasing(params::parameters, boundary_conditions)

    H_ZZ= params.J*two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)
    #display(L_D)
    L_D_dephasing = params.γ_d*(collective_Lindbladian_term(sp_sz, params))#+0.0001*one_body_Lindbladian_term(sp_sz, params))
    #display(L_D_dephasing)

    return L_H + L_D + L_D_dephasing
end

function sparse_DQIM_LRI(params::parameters, boundary_conditions)

    H_ZZ= params.J*LR_two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    H_Z = params.hz*one_body_Hamiltonian_term(params, sp_sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X + H_Z)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end

function sparse_DQIM_LRI_Schulz(params::parameters, boundary_conditions)

    z=0.5*(sp_id+sp_sz)

    H_ZZ= params.J*LR_two_body_Hamiltonian_term(params, z, z, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    H_Z = params.hz*one_body_Hamiltonian_term(params, z, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X + H_Z)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end

export sparse_2D_DQIM, sparse_Frustrated_DQIM

function sparse_2D_DQIM(params::parameters, boundary_conditions)

    H_ZZ= params.J*Ising_term_square(params, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    H_Z = params.hz*one_body_Hamiltonian_term(params, sp_sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X + H_Z)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end

function sparse_Frustrated_DQIM(params::parameters, boundary_conditions)

    H_ZZ= params.J*Ising_term_triangular(params, boundary_conditions)
    H_X = params.h*one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    H_Z = params.hz*one_body_Hamiltonian_term(params, sp_sz, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X + H_Z)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)

    return L_H + L_D
end



#### OLD: ####


export anisotropic_XY_local_dephasing, anisotropic_XY_collective_dephasing

function anisotropic_XY_local_dephasing(params::parameters, boundary_conditions)
    H_XX = params.J*two_body_Hamiltonian_term(params, sp_sx, sp_sx, boundary_conditions)
    H_YY =-params.J*two_body_Hamiltonian_term(params, sp_sy, sp_sy, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_XX + H_YY)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)
    L_D_dephasing = params.γ_d*one_body_Lindbladian_term(sp_sz, params)
    return L_H + L_D + L_D_dephasing
end

function anisotropic_XY_collective_dephasing(params::parameters, boundary_conditions)
    H_XX = params.J*two_body_Hamiltonian_term(params, sp_sx, sp_sx, boundary_conditions)
    H_YY =-params.J*two_body_Hamiltonian_term(params, sp_sy, sp_sy, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_XX + H_YY)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)
    L_D_dephasing = params.γ_d*collective_Lindbladian_term(sp_sz, params)
    return L_H + L_D + L_D_dephasing
end

export Bose_Hubbard, Bose_Hubbard_local_dephasing, Bose_Hubbard_collective_dephasing

function Bose_Hubbard(params::parameters, boundary_conditions)
    H_n = -params.h*one_body_Hamiltonian_term(params, sp_sp*sp_sm, boundary_conditions)
    H_driving = one_body_Hamiltonian_term(params, sp_sp+sp_sm, boundary_conditions)
    H_int = -params.J*(two_body_Hamiltonian_term(params, sp_sp, sp_sm, boundary_conditions) + two_body_Hamiltonian_term(params, sp_sm, sp_sp, boundary_conditions))
    L_H = vectorize_Hamiltonian(params, H_n + H_int + H_driving)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)
    return L_H + L_D
end

function Bose_Hubbard_local_dephasing(params::parameters, boundary_conditions)
    H_n = -params.h*one_body_Hamiltonian_term(params, sp_sp*sp_sm, boundary_conditions)
    H_driving = one_body_Hamiltonian_term(params, sp_sp+sp_sm, boundary_conditions)
    H_int = -params.J*(two_body_Hamiltonian_term(params, sp_sp, sp_sm, boundary_conditions) + two_body_Hamiltonian_term(params, sp_sm, sp_sp, boundary_conditions))
    L_H = vectorize_Hamiltonian(params, H_n + H_int + H_driving)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)
    L_D_dephasing = params.γ_d*one_body_Lindbladian_term(sp_sz, params)
    return L_H + L_D + L_D_dephasing
end

function Bose_Hubbard_collective_dephasing(params::parameters, boundary_conditions)
    H_n = -params.h*one_body_Hamiltonian_term(params, sp_sp*sp_sm, boundary_conditions)
    H_driving = one_body_Hamiltonian_term(params, sp_sp+sp_sm, boundary_conditions)
    H_int = -params.J*(two_body_Hamiltonian_term(params, sp_sp, sp_sm, boundary_conditions) + two_body_Hamiltonian_term(params, sp_sm, sp_sp, boundary_conditions))
    L_H = vectorize_Hamiltonian(params, H_n + H_int + H_driving)
    L_D = params.γ_l*one_body_Lindbladian_term(sp_sm, params)
    L_D_dephasing = params.γ_d*collective_Lindbladian_term(sp_sz, params)
    return L_H + L_D + L_D_dephasing
end




function DQIM_LRI(params::parameters, boundary_conditions)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D with long range coherent interactions
    """
    
    H_int = LR_two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_int + H_X)
    L_D = one_body_Lindbladian_term(params, sm, boundary_conditions)

    return L_H + L_D
end

function old_sparse_DQIM_LRI(params::parameters, boundary_conditions)
    
    H_int = LR_two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_int + H_X)
    L_D = one_body_Lindbladian_term(params, sp_sm, boundary_conditions)

    return L_H + L_D
end

function DQIM_LRL(params::parameters, boundary_conditions)#, N_K)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D with long range losses
    """

    H_ZZ= two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = LR_Lindbladian_term(params, sm, sp, boundary_conditions)

    return L_H + L_D
end

function sparse_DQIM_LRL(params::parameters, boundary_conditions)#, N_K)

    H_ZZ= two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = LR_Lindbladian_term(params, sp_sm, sp_sp, boundary_conditions)

    return L_H + L_D
end

function DQIM_LRD(params::parameters, boundary_conditions)#, N_K)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D with long range dephasing
    """

    H_ZZ= two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = LR_Lindbladian_term(params, sz, sz, boundary_conditions)

    return L_H + L_D
end

function sparse_DQIM_LRD(params::parameters, boundary_conditions)#, N_K)

    H_ZZ= two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = LR_Lindbladian_term(params, sp_sz, sp_sz, boundary_conditions)

    return L_H + L_D
end

function make_one_body_Lindbladian(H, Γ)
    L_H = -1im*(H⊗id - id⊗transpose(H))
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗id/2 - id⊗(transpose(Γ)*conj(Γ))/2
    return L_H + L_D
end