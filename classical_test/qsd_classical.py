import numpy as np
import scipy as sp
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed

def traj_evo(H:np.array, psi:np.array, dt:float, c_ops:list, evo_method:dict, rand_seed:int) -> np.array:
    """The single-step evolution of a single trajectory.
    
    Args: 
        H (np.array): The system Hamiltonian.
        psi (np.array): Wave function.
        dt (float): Time interval.
        c_ops (list): List of collapse operators.
        evo_method (dict): Information on evolutionary strategies.
        rand_seed (int): Random seed.

    Returns:
        The wave function of next step.
    """
    np.random.seed(rand_seed)
    p = 1000 # Approximation order
    k = len(c_ops) # Wiener increments number
    R = np.diag([1/(i+1) for i in range(p)])

    xis = np.random.normal(loc=0, scale=1, size=k)
    zetas = np.random.normal(loc=0, scale=1, size=(k,p))
    etas = np.random.normal(loc=0, scale=1, size=(k,p))
    mus = np.random.normal(loc=0, scale=1, size=k)
    phis = np.random.normal(loc=0, scale=1, size=k)

    rho_p = 1/12-np.sum([1/((i+1)**2) for i in range(p)])/(2*np.pi**2)
    alpha_p = np.pi**2/180-np.sum([1/((i+1)**4) for i in range(p)])/(2*np.pi**2)

    a0 = -(np.sqrt(2*dt)/np.pi)*np.sum([zetas[:,i]/(i+1) for i in range(p)],axis=0)-2*np.sqrt(dt*rho_p)*mus
    aij = (zetas@R@etas.T-etas@R@zetas.T)/(2*np.pi)
    c0 = (np.sqrt(2*dt)/(8*np.pi**3))*np.sum([zetas[:,i]/((i+1)**3) for i in range(p)],axis=0) # Fourth order

    c_expects = [0 for _ in c_ops]
    if evo_method['type'] == 'nonlinear':
        c_expects = [np.trace(np.outer(np.conj(psi), psi)@op) for op in c_ops]

    X_0 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+2*np.real(e_op)*op for op, e_op in zip(c_ops, c_expects)])
    H_eff = deepcopy(X_0*dt)
    if evo_method['order'] > 0:
        for i in range(k):
            H_eff += c_ops[i]*np.sqrt(dt)*xis[i]
            if evo_method['order'] > 1:
                _comm = X_0@c_ops[i]-c_ops[i]@X_0
                H_eff += _comm*a0[i]*dt/2
                for j in range(i+1,k):
                    _comm_ij = c_ops[i]@c_ops[j]-c_ops[j]@c_ops[i]
                    if np.sum(np.abs(_comm_ij)) != 0:
                        H_eff += 0.5*_comm_ij*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])
                if evo_method['order'] > 2:
                    _comm = X_0@_comm-_comm@X_0
                    H_eff += _comm*(dt**2)*(np.sqrt(dt*alpha_p)*phis[i]+np.sqrt(dt/2)*np.sum([etas[i,l]/(l+1)**2 for l in range(p)])/np.pi)/(2*np.pi)
                    if evo_method['order'] > 3:
                        _comm = X_0@_comm-_comm@X_0
                        H_eff += _comm*dt*dt*dt*c0[i]
    else:
        for i in range(k):
            H_eff += c_ops[i]*np.sqrt(dt)*xis[i]
        H_eff -= dt*sum([-0.5*op@op+2*np.real(e_op)*op for op, e_op in zip(c_ops, c_expects[0])]) # Euler-Maruyama

    psi_p = sp.sparse.linalg.expm_multiply(H_eff, psi)

    if evo_method['nonlinear_corr']:
        c_expects_p = [np.trace(np.outer(np.conj(psi_p), psi_p)@op) for op in c_ops]
        X_0 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+np.real(e_op+e_op_p)*op for op, e_op, e_op_p in zip(c_ops, c_expects, c_expects_p)])
        H_eff = deepcopy(X_0*dt)
        if evo_method['order'] > 0:
            for i in range(k):
                H_eff += c_ops[i]*np.sqrt(dt)*xis[i]
                if evo_method['order'] > 1:
                    _comm = X_0@c_ops[i]-c_ops[i]@X_0
                    H_eff += _comm*a0[i]*dt/2
                    for j in range(i+1,k):
                        _comm_ij = c_ops[i]@c_ops[j]-c_ops[j]@c_ops[i]
                        if np.sum(np.abs(_comm_ij)) != 0:
                            H_eff += 0.5*_comm_ij*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])
                    if evo_method['order'] > 2:
                        _comm = X_0@_comm-_comm@X_0
                        H_eff += _comm*(dt**2)*(np.sqrt(dt*alpha_p)*phis[i]+np.sqrt(dt/2)*np.sum([etas[i,l]/(l+1)**2 for l in range(p)])/np.pi)/(2*np.pi)
                        if evo_method['order'] > 3:
                            _comm = X_0@_comm-_comm@X_0
                            H_eff += _comm*dt*dt*dt*c0[i]
        else:
            for i in range(k):
                H_eff += c_ops[i]*np.sqrt(dt)*xis[i]
            H_eff -= dt*sum([-0.5*op@op+2*np.real(e_op)*op for op, e_op in zip(c_ops, c_expects[0])]) # Euler-Maruyama

        psi_p = sp.sparse.linalg.expm_multiply(H_eff, psi)

    if evo_method['type'] == 'nonlinear':
        psi_p = psi_p/np.linalg.norm(psi_p)

    return psi_p

def traj_evo_comparison(H:np.array, psis:list, dt:float, c_ops:list, evo_method:dict, rand_seed:int) -> tuple:
    """The single-step evolution of a single trajectory, comparison of the 1,2,3,4 order schemes.
    
    Args: 
        H (np.array): The system Hamiltonian.
        psis (list): Wave functions.
        dt (float): Time interval.
        c_ops (list): List of collapse operators.
        evo_method (dict): Information on evolutionary strategies.
        rand_seed (int): Random seed.

    Returns:
        The wave function of next step.
    """
    np.random.seed(rand_seed)
    p = 1000 # Approximation order
    k = len(c_ops) # Wiener increments number
    R = np.diag([1/(i+1) for i in range(p)])

    xis = np.random.normal(loc=0, scale=1, size=k)
    zetas = np.random.normal(loc=0, scale=1, size=(k,p))
    etas = np.random.normal(loc=0, scale=1, size=(k,p))
    mus = np.random.normal(loc=0, scale=1, size=k)
    phis = np.random.normal(loc=0, scale=1, size=k)

    rho_p = 1/12-np.sum([1/((i+1)**2) for i in range(p)])/(2*np.pi**2)
    alpha_p = np.pi**2/180-np.sum([1/((i+1)**4) for i in range(p)])/(2*np.pi**2)

    a0 = -(np.sqrt(2*dt)/np.pi)*np.sum([zetas[:,i]/(i+1) for i in range(p)],axis=0)-2*np.sqrt(dt*rho_p)*mus
    aij = (zetas@R@etas.T-etas@R@zetas.T)/(2*np.pi)
    c0 = (np.sqrt(2*dt)/(8*np.pi**3))*np.sum([zetas[:,i]/((i+1)**3) for i in range(p)],axis=0)

    c_expects = [[0 for _ in c_ops] for i in range(4)]
    if evo_method['type'] == 'nonlinear':
        c_expects = [[np.trace(np.outer(np.conj(psis[i]), psis[i])@op) for op in c_ops] for i in range(4)]

    # unified random number sequence and comparison of the 1,2,3,4 order schemes
    X_01 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+2*np.real(e_op)*op for op, e_op in zip(c_ops, c_expects[0])])
    X_02 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+2*np.real(e_op)*op for op, e_op in zip(c_ops, c_expects[1])])
    X_03 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+2*np.real(e_op)*op for op, e_op in zip(c_ops, c_expects[2])])
    X_04 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+2*np.real(e_op)*op for op, e_op in zip(c_ops, c_expects[3])])
    H_eff1, H_eff2, H_eff3, H_eff4 = deepcopy(X_01*dt), deepcopy(X_02*dt), deepcopy(X_03*dt), deepcopy(X_04*dt)
    psi1, psi2, psi3, psi4 = deepcopy(psis[0]), deepcopy(psis[1]), deepcopy(psis[2]), deepcopy(psis[3])

    for i in range(k):
        H_eff1 += c_ops[i]*np.sqrt(dt)*xis[i]
        _comm2 = X_02@c_ops[i]-c_ops[i]@X_02
        _comm3 = X_03@c_ops[i]-c_ops[i]@X_03
        _comm4 = X_04@c_ops[i]-c_ops[i]@X_04
        H_eff2 += c_ops[i]*np.sqrt(dt)*xis[i]+_comm2*a0[i]*dt/2
        H_eff3 += c_ops[i]*np.sqrt(dt)*xis[i]+_comm3*a0[i]*dt/2
        H_eff4 += c_ops[i]*np.sqrt(dt)*xis[i]+_comm4*a0[i]*dt/2
        _comm3 = X_03@_comm3-_comm3@X_03
        _comm4 = X_04@_comm4-_comm4@X_04
        H_eff3 += _comm3*(dt**2)*(np.sqrt(dt*alpha_p)*phis[i]+np.sqrt(dt/2)*np.sum([etas[i,l]/(l+1)**2 for l in range(p)])/np.pi)/(2*np.pi)
        H_eff4 += _comm4*(dt**2)*(np.sqrt(dt*alpha_p)*phis[i]+np.sqrt(dt/2)*np.sum([etas[i,l]/(l+1)**2 for l in range(p)])/np.pi)/(2*np.pi)
        _comm4 = X_04@_comm4-_comm4@X_04
        H_eff4 += _comm4*dt*dt*dt*c0[i]
        for j in range(i+1,k):
            _comm = c_ops[i]@c_ops[j]-c_ops[j]@c_ops[i]
            if np.sum(np.abs(_comm)) != 0:
                H_eff2 += 0.5*_comm*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])
                H_eff3 += 0.5*_comm*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])
                H_eff4 += 0.5*_comm*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])

    psi1_p = sp.sparse.linalg.expm_multiply(H_eff1, psi1)
    psi2_p = sp.sparse.linalg.expm_multiply(H_eff2, psi2)
    psi3_p = sp.sparse.linalg.expm_multiply(H_eff3, psi3)
    psi4_p = sp.sparse.linalg.expm_multiply(H_eff4, psi4)
    psis_p = [psi1_p, psi2_p, psi3_p, psi4_p]

    if evo_method['nonlinear_corr']:
        c_expects_p = [[np.trace(np.outer(np.conj(psis_p[i]), psis_p[i])@op) for op in c_ops] for i in range(4)]
        X_01 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+np.real(e_op+e_op_p)*op for op, e_op, e_op_p in zip(c_ops, c_expects[0], c_expects_p[0])])
        X_02 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+np.real(e_op+e_op_p)*op for op, e_op, e_op_p in zip(c_ops, c_expects[1], c_expects_p[0])])
        X_03 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+np.real(e_op+e_op_p)*op for op, e_op, e_op_p in zip(c_ops, c_expects[2], c_expects_p[0])])
        X_04 = -1J*H+sum([-0.5*(np.conj(op).T+op)@op+np.real(e_op+e_op_p)*op for op, e_op, e_op_p in zip(c_ops, c_expects[3], c_expects_p[0])])
        H_eff1, H_eff2, H_eff3, H_eff4 = deepcopy(X_01*dt), deepcopy(X_02*dt), deepcopy(X_03*dt), deepcopy(X_04*dt)
        psi1, psi2, psi3, psi4 = deepcopy(psis[0]), deepcopy(psis[1]), deepcopy(psis[2]), deepcopy(psis[3])

        for i in range(k):
            H_eff1 += c_ops[i]*np.sqrt(dt)*xis[i]
            _comm2 = X_02@c_ops[i]-c_ops[i]@X_02
            _comm3 = X_03@c_ops[i]-c_ops[i]@X_03
            _comm4 = X_04@c_ops[i]-c_ops[i]@X_04
            H_eff2 += c_ops[i]*np.sqrt(dt)*xis[i]+_comm2*a0[i]*dt/2
            H_eff3 += c_ops[i]*np.sqrt(dt)*xis[i]+_comm3*a0[i]*dt/2
            H_eff4 += c_ops[i]*np.sqrt(dt)*xis[i]+_comm4*a0[i]*dt/2
            _comm3 = X_03@_comm3-_comm3@X_03
            _comm4 = X_04@_comm4-_comm4@X_04
            H_eff3 += _comm3*(dt**2)*(np.sqrt(dt*alpha_p)*phis[i]+np.sqrt(dt/2)*np.sum([etas[i,l]/(l+1)**2 for l in range(p)])/np.pi)/(2*np.pi)
            H_eff4 += _comm4*(dt**2)*(np.sqrt(dt*alpha_p)*phis[i]+np.sqrt(dt/2)*np.sum([etas[i,l]/(l+1)**2 for l in range(p)])/np.pi)/(2*np.pi)
            _comm4 = X_04@_comm4-_comm4@X_04
            H_eff4 += _comm4*dt*dt*dt*c0[i]
            for j in range(i+1,k):
                _comm = c_ops[i]@c_ops[j]-c_ops[j]@c_ops[i]
                if np.sum(np.abs(_comm)) != 0:
                    H_eff2 += 0.5*_comm*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])
                    H_eff3 += 0.5*_comm*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])
                    H_eff4 += 0.5*_comm*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])

        psi1_p = sp.sparse.linalg.expm_multiply(H_eff1, psi1)
        psi2_p = sp.sparse.linalg.expm_multiply(H_eff2, psi2)
        psi3_p = sp.sparse.linalg.expm_multiply(H_eff3, psi3)
        psi4_p = sp.sparse.linalg.expm_multiply(H_eff4, psi4)
        psis_p = [psi1_p, psi2_p, psi3_p, psi4_p]

    if evo_method['type'] == 'nonlinear':
        psis_p = [wf/np.linalg.norm(wf) for wf in psis_p]
    
    return psis_p
    
def qsdsolve(H:np.array, initial:list, tlist:np.array, c_ops:list, e_ops:list, evo_method:dict) -> tuple:
    """QSD solver, comparison of the 1,2,3,4 order schemes.
    
    Args: 
        H (np.array): The system Hamiltonian.
        initial (list): Initial state.
        tlist (np.array): Time array.
        c_ops (list): List of collapse operators.
        e_ops (list): List of observables.
        evo_method (dict): Information on evolutionary strategies.

    Returns:
        Expectation values during evolution.
    """
    e_list, v_list = [], []
    esb = initial
    period = evo_method['period']

    for i in tqdm(range(len(tlist))):
        expect, var = [], []
        for op in e_ops:
            if evo_method['comparison']:
                _temps = [np.asarray([np.trace(np.outer(np.conj(wf[j]), wf[j])@op) for wf in esb]) for j in range(4)]
                expect.append([np.real(np.mean(_temps[j])) for j in range(4)])
                var.append([np.real(np.std(_temps[j])) for j in range(4)])
            else:
                _temps = np.asarray([np.trace(np.outer(np.conj(wf), wf)@op) for wf in esb])
                expect.append(np.real(np.mean(_temps)))
                var.append(np.real(np.std(_temps)))

        e_list.append(expect)
        v_list.append(var)
        if i < len(tlist)-1:
        # add parallelization using joblib
            if evo_method['comparison']:
                esb = Parallel(n_jobs=-1)(delayed(traj_evo_comparison)(
                    H, psis, tlist[i+1]-tlist[i], c_ops, evo_method, (i)+k*period
                    ) for k, psis in enumerate(esb))
            else:
                esb = Parallel(n_jobs=-1)(delayed(traj_evo)(
                    H, psi, tlist[i+1]-tlist[i], c_ops, evo_method, (i)+k*period
                    ) for k, psi in enumerate(esb))
    
    return e_list, v_list
