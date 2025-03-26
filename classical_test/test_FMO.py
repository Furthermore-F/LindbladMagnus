import matplotlib.pyplot as plt 
import numpy as np
import scipy.constants as C
from copy import deepcopy
from qutip import *

from constants import *
from qsd_classical import qsdsolve

# Evolution parameters
step_num = 100
T = 500
tlist = np.linspace(0.0, T, step_num+1)
traj_num = 1
evo_method = {'period': 42424, 
              'type': 'nonlinear', 
              'nonlinear_corr': False,
              'comparison': False,
              'order': 1}

# Model parameters
a = 3*1e-3
b = 5*1e-7
g = 6.28*1e-3

# Hamiltonian
H = np.array([[0, 0, 0, 0, 0],
              [0, 0.0267, -0.0129, 0.000632, 0],
              [0, -0.0129, 0.0273, 0.00404, 0],
              [0, 0.000632, 0.00404, 0, 0],
              [0, 0, 0, 0, 0]])*1.6022/1.05457266+0J

# Lindblad operators and initial mnbd operator
site_1 = np.array([[0], [1], [0], [0], [0]])+0j
site_2 = np.array([[0], [0], [1], [0], [0]])+0j
site_3 = np.array([[0], [0], [0], [1], [0]])+0j
sink = np.array([[0], [0], [0], [0], [1]])+0j
ground = np.array([[1], [0], [0], [0], [0]])+0j

L_deph_1 = np.sqrt(a)*np.outer(site_1, site_1)
L_deph_2 = np.sqrt(a)*np.outer(site_2, site_2)
L_deph_3 = np.sqrt(a)*np.outer(site_3, site_3)
L_diss_1 = np.sqrt(b)*np.outer(ground, site_1)
L_diss_2 = np.sqrt(b)*np.outer(ground, site_2)
L_diss_3 = np.sqrt(b)*np.outer(ground, site_3)
L_sink = np.sqrt(g)*np.outer(sink, site_3)

c_ops = [L_deph_1, L_deph_2, L_deph_3, L_diss_1, L_diss_2, L_diss_3, L_sink]
e_ops = [np.outer(site_1, site_1), 
         np.outer(site_2, site_2), 
         np.outer(site_3, site_3),
         np.outer(sink, sink), 
         np.outer(ground, ground)]
e_ops_name = ['Site 1', 'Site 2', 'Site 3', 'Sink', 'Ground']
psi0 = np.array([[0], [1], [0], [0], [0]])+0j

# Plot exact and QSD results
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
ax1.set_xlabel('Time')
ax1.set_ylabel('Population')
ax1.set_xlim([0,T])
ax2.set_xlabel('Time')
ax2.set_ylabel('Population Error')
ax2.set_xlim([0,T])

# Compare QSD results and Mesolve exact results
result = mesolve(Qobj(H), Qobj(psi0), tlist, [Qobj(op) for op in c_ops], [Qobj(op) for op in e_ops])
for i in range(len(e_ops)):
    ax1.plot(tlist, result.expect[i], label = f'Exact {e_ops_name[i]}')
if evo_method['comparison']:
    file_name = f'classical_test/FMO/comparison_{evo_method["type"]}_{traj_num}traj_{step_num}step_{evo_method["nonlinear_corr"]}_{evo_method["period"]}'
    initial = [[psi0]*4]*traj_num # initial wavefunction ensemble
    e_list, _ = qsdsolve(H, initial, tlist, c_ops, e_ops, evo_method)
    for stat in range(len(e_ops)):
        for order in range(1):
            ax1.plot(tlist, [e_list[i][stat][order] for i in range(len(tlist))], 
                    label = f'QSD {e_ops_name[stat]}, M{order+1}')
            ax2.plot(tlist, [np.abs(e_list[i][stat][order]-result.expect[stat][i]) for i in range(len(tlist))], 
                    label = f'QSD {e_ops_name[stat]}, M{order+1}')
else:
    order = evo_method['order']
    file_name = f'classical_test/FMO/{order}order_{evo_method["type"]}_{traj_num}traj_{step_num}step_{evo_method["nonlinear_corr"]}_{evo_method["period"]}'
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
plt.show()
