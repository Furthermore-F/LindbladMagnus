import matplotlib.pyplot as plt 
import numpy as np
import scipy.constants as C
from copy import deepcopy
from qutip import *

from constants import *
from qsd_classical import qsdsolve

# Evolution parameters
step_num = 100
T = 25
tlist = np.linspace(0.0, T, step_num+1)
traj_num = 1000
evo_method = {'period': 42424, 
              'type': 'linear', 
              'nonlinear_corr': False,
              'comparison': False,
              'order':1 # Zero order for EM method
              }

# Hamiltonian
g = 0.1
H = np.kron(PZ,PZ)-0.5*(np.kron(PX,ID) + np.kron(ID,PX))

# Lindblad operators and initial mnbd operator
op = np.sqrt(g)*PL
c_ops = [np.kron(op, ID), np.kron(ID, op)]
e_ops = [np.kron(PJ0, PJ0), np.kron(PJ1, PJ1), np.kron(PJ0, PJ1)]
e_ops_name = ['|00>', '|11>', '|01>']
psi0 = np.array([[0], [0], [0], [1]])+0j

# Plot exact and QSD results
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
ax1.set_xlabel('Time')
ax1.set_ylabel('Population')
ax1.set_xlim([0,T])
ax2.set_xlabel('Time')
ax2.set_ylabel('Population Error')
ax2.set_xlim([0,T])

tlist_exact = np.linspace(0.0, T, step_num*100+1)
result = mesolve(Qobj(H), Qobj(psi0), tlist_exact, [Qobj(op) for op in c_ops], [Qobj(op) for op in e_ops])
ax1.plot(tlist_exact, result.expect[0], label = 'Exact |00>')
ax1.plot(tlist_exact, result.expect[1], label = 'Exact |11>')
ax1.plot(tlist_exact, result.expect[2], label = 'Exact |01>')

# Compare QSD results and Mesolve exact results
result = mesolve(Qobj(H), Qobj(psi0), tlist, [Qobj(op) for op in c_ops], [Qobj(op) for op in e_ops])
if evo_method['comparison']:
    file_name = f'classical_test/TFIM/comparison_{evo_method["type"]}_{traj_num}traj_{step_num}step_{evo_method["nonlinear_corr"]}_{evo_method["period"]}'
    initial = [[psi0]*4]*traj_num # initial wavefunction ensemble
    e_list, _ = qsdsolve(H, initial, tlist, c_ops, e_ops, evo_method)
    for stat in range(len(e_ops)):
        for order in range(2):
            ax1.scatter(tlist, [e_list[i][stat][order] for i in range(len(tlist))], 
                        label = f'QSD {e_ops_name[stat]}, M{order+1}')
            ax2.plot(tlist, [np.abs(e_list[i][stat][order]-result.expect[stat][i]) for i in range(len(tlist))], 
                    label = f'QSD {e_ops_name[stat]}, M{order+1}')
else:
    order = evo_method['order']
    file_name = f'classical_test/TFIM/{order}order_{evo_method["type"]}_{traj_num}traj_{step_num}step_{evo_method["nonlinear_corr"]}_{evo_method["period"]}'
    initial = [psi0]*traj_num # initial wavefunction ensemble
    e_list, _ = qsdsolve(H, initial, tlist, c_ops, e_ops, evo_method)
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