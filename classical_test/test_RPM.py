import matplotlib.pyplot as plt 
import numpy as np
import scipy.constants as C
from copy import deepcopy
from qutip import *

from constants import *
from qsd_classical import qsdsolve

# Evolution parameters
step_num = 100
T = 1e-5
tlist = np.linspace(0.0, T, step_num+1)
traj_num = 1000
evo_method = {'period': 42424, 
              'type': 'linear', 
              'nonlinear_corr': False,
              'comparison': True,
              'order': 4}

# Model parameters
A = 1e-4*np.array([0.345, 0.345, 9])+0j
B_0 = 47 # in uT
g = 0.5*C.e/C.electron_mass
k_d = 1e2*np.sqrt(2)
theta_angle = 90

# Hamiltonian
theta = np.pi*theta_angle/180
B = B_0*1e-6*np.array([np.sin(theta), 0, np.cos(theta)])+0j
H = 0
for i, op in enumerate(PS):
    H += g*A[i]*np.kron(np.kron(op, op), ID)*0.25
    H += g*B[i]*(np.kron(np.kron(ID, op), ID) + np.kron(np.kron(ID, ID), op))
H = np.kron(H, PJ0)

# Lindblad operators and initial mnbd operator
state_1e = [np.array([[1], [0]])+0j, np.array([[0], [1]])+0j]
state_2e = [1/np.sqrt(2)*(np.kron(state_1e[0], state_1e[1])-np.kron(state_1e[1], state_1e[0])), # singlet state
            1/np.sqrt(2)*(np.kron(state_1e[0], state_1e[1])+np.kron(state_1e[1], state_1e[0])), # triplet state t_0
            np.kron(state_1e[0], state_1e[0]), # triplet state t_p
            np.kron(state_1e[1], state_1e[1]), # triplet state t_m
            ]
shelf_s = np.kron(np.kron(state_1e[0], state_2e[2]), state_1e[1]) # singlet shelf
shelf_t = np.kron(np.kron(state_1e[1], state_2e[3]), state_1e[1]) # triplet shelf

c_ops = []
for op_1 in state_1e:
    for i, op_2 in enumerate(state_2e):
        op = shelf_t
        if i == 0: 
            op = shelf_s
        c_ops.append(np.outer(op, np.kron(np.kron(op_1, op_2), state_1e[0]))*k_d)
e_ops = [np.outer(shelf_s, shelf_s), np.outer(shelf_t, shelf_t)]
e_ops_name = ['Singlet', 'Triplet']
rho0 = np.kron(ID, np.outer(np.kron(state_2e[0], state_1e[0]), np.kron(state_2e[0], state_1e[0])))
_psi0 = np.kron(np.kron(state_1e[0], state_2e[0]), state_1e[0])
_psi1 = np.kron(np.kron(state_1e[1], state_2e[0]), state_1e[0])

# Plot exact and QSD results
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.set_xlabel('Time')
ax1.set_ylabel('Yield')
ax1.set_xlim([0,T])
ax2.set_xlabel('Time')
ax2.set_ylabel('Yield Error')
ax2.set_xlim([0,T])

# Compare QSD results and Mesolve exact results
result = mesolve(Qobj(H), Qobj(rho0).unit(), tlist, [Qobj(op) for op in c_ops], [Qobj(op) for op in e_ops])
ax1.plot(tlist, result.expect[0], label = 'Exact Singlet')
ax1.plot(tlist, result.expect[1], label = 'Exact Triplet')
if evo_method['comparison']:
    file_name = f'classical_test/RPM/comparison_{evo_method["type"]}_{traj_num}traj_{step_num}step_{B_0}T_{theta_angle}d'
    initial = [[_psi0]*4]*(traj_num//2) + [[_psi1]*4]*(traj_num//2) # initial wavefunction ensemble
    e_list, _ = qsdsolve(H, initial, tlist, c_ops, e_ops, evo_method)
    for stat in range(len(e_ops)):
        for order in range(4):
            ax1.plot(tlist, [e_list[i][stat][order] for i in range(len(tlist))], 
                    label = f'QSD {e_ops_name[stat]}, M{order+1}')
            ax2.plot(tlist, [np.abs(e_list[i][stat][order]-result.expect[stat][i]) for i in range(len(tlist))], 
                    label = f'QSD {e_ops_name[stat]}, M{order+1}')
else:
    order = evo_method['order']
    file_name = f'classical_test/RPM/{order}order_{evo_method["type"]}_{traj_num}traj_{step_num}step_{B_0}T_{theta_angle}d'
    initial = [_psi0]*(traj_num//2) + [_psi1]*(traj_num//2) # initial wavefunction ensemble
    _, e_list = qsdsolve(H, initial, tlist, c_ops, e_ops, evo_method)
    for stat in range(len(e_ops)):
        ax1.plot(tlist, [e_list[i][stat] for i in range(len(tlist))], 
                 label = f'QSD {e_ops_name[stat]}, M{order}')
        ax2.plot(tlist, [np.abs(e_list[i][stat]-result.expect[stat][i]) for i in range(len(tlist))], 
                 label = f'QSD {e_ops_name[stat]}, M{order}')

# Save data and figures
np.savez(f'{file_name}.npz', exact_result = result.expect, qsd_result = e_list)
ax1.legend()
ax2.legend()
plt.savefig(f'{file_name}.png', dpi=600)
plt.show()
