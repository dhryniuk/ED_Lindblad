export DQIM, sparse_DQIM, LRInt_DQIM, sparse_LRInt_DQIM, LRDisp_DQIM, sparse_LRDisp_DQIM


function DQIM(params::parameters, boundary_conditions)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D
    """

    H_ZZ= two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = one_body_Lindbladian_term(params, sm, boundary_conditions)

    return L_H + L_D
end

function sparse_DQIM(params::parameters, boundary_conditions)

    H_ZZ= two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = one_body_Lindbladian_term(params, sp_sm, boundary_conditions)

    return L_H + L_D
end

function LRInt_DQIM(params::parameters, boundary_conditions)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D with long range coherent interactions
    """
    
    H_int = LR_two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_int + H_X)
    L_D = one_body_Lindbladian_term(params, sm, boundary_conditions)

    return L_H + L_D
end

function sparse_LRInt_DQIM(params::parameters, boundary_conditions)
    
    H_int = LR_two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_int + H_X)
    L_D = one_body_Lindbladian_term(params, sp_sm, boundary_conditions)

    return L_H + L_D
end

function LRDisp_DQIM(params::parameters, boundary_conditions)#, N_K)
    """
    Returns the matrix Lindblad superoperator for the dissipative transverse field Ising model in 1D with long range dissipative interactions
    """

    H_ZZ= two_body_Hamiltonian_term(params, sz, sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = LR_Lindbladian_term(params, sm, sp, boundary_conditions)

    return L_H + L_D
end

function sparse_LRDisp_DQIM(params::parameters, boundary_conditions)#, N_K)

    H_ZZ= two_body_Hamiltonian_term(params, sp_sz, sp_sz, boundary_conditions)
    H_X = one_body_Hamiltonian_term(params, sp_sx, boundary_conditions)
    L_H = vectorize_Hamiltonian(params, H_ZZ + H_X)
    L_D = LR_Lindbladian_term(params, sp_sm, sp_sp, boundary_conditions)

    return L_H + L_D
end