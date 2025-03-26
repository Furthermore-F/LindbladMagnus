from joblib import Parallel, delayed
from qutip import *
import argparse
import time, copy
import scipy.constants as C
import numpy as np
import matplotlib.pyplot as plt

from qsd_quantum import single_traj_evo
from utils import *

parser = argparse.ArgumentParser(description='Lindblad Dynamics')

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
rho0 = np.kron(ID, np.outer(np.kron(state_2e[0], state_1e[0]), np.kron(state_2e[0], state_1e[0])))
_psi0 = np.kron(np.kron(state_1e[0], state_2e[0]), state_1e[0])
_psi1 = np.kron(np.kron(state_1e[1], state_2e[0]), state_1e[0])

# Hamiltonian Information
parser.add_argument('--system_name', type=str, help='Name of physical/chemical systems.', default='RPM')
parser.add_argument('--hamiltonian', type=np.array, help='System Hamiltonian.', default=H)
parser.add_argument('--channels', type=list, help='Operators of damping channels.', default=c_ops)
parser.add_argument('--magnus_order', type=int, help='Order of Magnus expansion.', default=1)
parser.add_argument('--types', type=str, help='The type of QSD (linear or nonlinear).', default='Linear')
parser.add_argument('--observation', type=list, help='The operators that need to be measured.', default=e_ops)

# Circuit Information
parser.add_argument('--param_num_per_layer', type=int, help='Name of parameters per layer.', default=41)
parser.add_argument('--layer_number', type=int, help='Number of layers.', default=2)
parser.add_argument('--qubit_num', type=int, help='Number of qubits.', default=5)
parser.add_argument('--bit_num', type=int, help='Number of classical bits.', default=0)
parser.add_argument('--initialize', type=dict, help='The initial state.', default=np.kron(_psi0, np.asarray([[1], [0]])).flatten())
parser.add_argument('--unit_str', type=str, help='The quantum circuits for each layer.',
                    default=' RX(p{},1); RY(p{},1); RZ(p{},1); RX(p{},3); RY(p{},3); RZ(p{},3); CX(1,2); RX(p{},2); RY(p{},2); RZ(p{},2); CX(1,2);'
                    ' CX(3,4);CX(1,3); RX(p{},3); RY(p{},3); RZ(p{},3); CX(1,3); RX(p{},4); RY(p{},4); RZ(p{},4); CX(3,4); '
                    'CX(4,3); CX(3,1); RX(p{},1);'
                    'RY(p{},1); RZ(p{},1); CX(3,1); CX(4,3); CX(2,4); RZ(p{},4); CX(2,4); CX(1,3); RZ(p{},3); '
                    'CX(1,3); RX(p{},2); RY(p{},2); RZ(p{},2); CX(4,3); CX(3,1);'
                    'RX(p{},1); RY(p{},1); RZ(p{},1); CX(3,1); CX(4,3); CX(1,4); RZ(p{},1); CX(1,4); CX(3,4); RZ(p{},4); CX(3,4);'
                    'CX(1,4); RZ(p{},4); CX(1,4); CX(4,2); CX(2,1); RX(p{},1); RY(p{},1); RZ(p{},1); CX(2,1);'
                    'CX(4,2); CX(1,3); RZ(p{},3); CX(1,3); CX(4,3); CX(3,2); CX(2,1); RX(p{},1); RY(p{},1);'
                    'RZ(p{},1); CX(2,1); CX(3,2); CX(4,3); CX(1,4); RZ(p{},4); CX(1,4); CX(1,2); RX(p{},2);'
                    'CX(3,1); RX(p{},1); RY(p{},1); RZ(p{},1); CX(3,1);')

# Evolution Information
parser.add_argument('--method', type=str, help='Method choice of evolution, QSD or QJ.', default='QSD')
parser.add_argument('--simulator', type=str, help='Name of simulator.', default='statevector')
parser.add_argument('--device', type=str, help='Simulator device.', default='CPU')
parser.add_argument('--random_seed', type=int, help='Seed of random numbers.', default=42424)
parser.add_argument('--traj_num', type=int, help='Number of trajectories.', default=1)
parser.add_argument('--time_interval', type=float, help='Length of time steps.', default=5e-8)
parser.add_argument('--step_num', type=int, help='Number of time steps.', default=101)
parser.add_argument('--parallel', type=int, help='Number of processes in Joblib.', default=1)

# Initialize
args = parser.parse_args()
hamil_info, circ_info, evo_info = initialize_info(args)
file_info = f'quantum_test/{args.system_name}/{args.traj_num}traj_{args.step_num}step_{args.types}_{args.magnus_order}'
print_info(args)

fig_evo, ax_evo = plt.subplots()
fig_err, ax_err = plt.subplots()

# Exact Results
times = np.linspace(0, (evo_info['step_num']-1)*evo_info['time_interval'], evo_info['step_num'])
exact_results = mesolve(Qobj(H), Qobj(rho0).unit(), times, [Qobj(op) for op in c_ops], [Qobj(op) for op in e_ops]).expect

# Variational Quantum Simulation 
t = time.perf_counter()
results = Parallel(n_jobs=args.parallel, backend="loky")(
    delayed(single_traj_evo)(circ_info=circ_info, 
                             evo_info=evo_info, 
                             hamil_info=hamil_info,
                             random_seed=i*args.random_seed) for i in range(evo_info['traj_num'])
    )
print(f'Total time cost: {time.perf_counter()-t:.4f}')
traj_averages = np.mean(results, axis=0)

# Plot & Save
markers=['', '']
linestyles=['-', '-']
labels=["Exact Singlet", "Exact Triplet"]
plot_evo_curve(evo_data=exact_results, ax=ax_evo, times=times, 
               markers=markers, linestyles=linestyles, labels=labels, 
               xlabel='Time', ylabel='Yield')

linestyles=['--', '--']
labels=["QSD Singlet", "QSD Triplet"]
plot_evo_curve(evo_data=np.asarray(traj_averages).T, ax=ax_evo, times=times, 
               markers=markers, linestyles=linestyles, labels=labels, 
               xlabel='Time', ylabel='Yield')
plot_err_curve(evo_data=np.asarray(traj_averages).T, ref_data=np.asarray(exact_results), ax=ax_err, times=times, 
               markers=markers, linestyles=linestyles, labels=labels, 
               xlabel='Time', ylabel='Yield')

ax_evo.legend()
ax_err.legend()
fig_evo.savefig(f'{file_info}_evo', dpi=600)
fig_err.savefig(f'{file_info}_err.png', dpi=600)
np.savez(f'{file_info}_data.npz', results=results)
