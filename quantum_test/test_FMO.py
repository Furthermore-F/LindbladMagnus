from joblib import Parallel, delayed
from qutip import *
import argparse
import time, copy
import numpy as np
import matplotlib.pyplot as plt

from qsd_quantum import single_traj_evo
from utils import *

parser = argparse.ArgumentParser(description='Lindblad Dynamics')

a = 3*1e-3
b = 5*1e-7
g = 6.28*1e-3

H = np.array([[0, 0, 0, 0, 0],
              [0, 0.0267, -0.0129, 0.000632, 0],
              [0, -0.0129, 0.0273, 0.00404, 0],
              [0, 0.000632, 0.00404, 0, 0],
              [0, 0, 0, 0, 0]])*1.6022/1.05457266+0J
H = np.pad(H, ((0, 3), (0, 3)), 'constant', constant_values=0)

ground = np.array([[1], [0], [0], [0], [0], [0], [0], [0]])+0j
site_1 = np.array([[0], [1], [0], [0], [0], [0], [0], [0]])+0j
site_2 = np.array([[0], [0], [1], [0], [0], [0], [0], [0]])+0j
site_3 = np.array([[0], [0], [0], [1], [0], [0], [0], [0]])+0j
sink = np.array([[0], [0], [0], [0], [1], [0], [0], [0]])+0j

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
psi0 = np.array([[0], [1], [0], [0], [0], [0], [0], [0]])+0j

# Hamiltonian Information
parser.add_argument('--system_name', type=str, help='Name of physical/chemical systems.', default='FMO')
parser.add_argument('--hamiltonian', type=np.array, help='System Hamiltonian.', default=H)
parser.add_argument('--channels', type=list, help='Operators of damping channels.', default=c_ops)
parser.add_argument('--magnus_order', type=int, help='Order of Magnus expansion.', default=1)
parser.add_argument('--types', type=str, help='The type of QSD (linear or nonlinear).', default='Nonlinear')
parser.add_argument('--observation', type=list, help='The operators that need to be measured.', default=e_ops)

# Circuit Information
parser.add_argument('--param_num_per_layer', type=int, help='Name of parameters per layer.', default=16)
parser.add_argument('--layer_number', type=int, help='Number of layers.', default=1)
parser.add_argument('--qubit_num', type=int, help='Number of qubits.', default=4)
parser.add_argument('--bit_num', type=int, help='Number of classical bits.', default=0)
parser.add_argument('--initialize', type=dict, help='The initial state.', default=np.kron(psi0, np.asarray([[1], [0]])).flatten())
parser.add_argument('--unit_str', type=str, help='The quantum circuits for each layer.',
                    default=' RX(p{},1); RY(p{},1); RZ(p{},3); RZ(p{},1); H(1); H(2); CX(1,2); RZ(p{},2); RX(p{},2); '
                    'RY(p{},2); CX(1,2); H(1); CX(1,2); RZ(p{},2); CX(1,2); H(2); CX(1,2); RZ(p{},2); CX(1,2); CX(3,1); '
                    'RZ(p{},1); CX(3,1); CX(1,3); RZ(p{},3); CX(1,3); CX(3,2); RZ(p{},2); CX(3,2); CX(3,2); CX(2,1); RZ(p{},1); '
                    'RX(p{},3); RY(p{},3); RZ(p{},3); CX(2,1); CX(3,2);')

# Evolution Information
parser.add_argument('--method', type=str, help='Method choice of evolution, QSD or QJ.', default='QSD')
parser.add_argument('--simulator', type=str, help='Name of simulator.', default='statevector')
parser.add_argument('--noisy_simulation', type=bool, help='Whether or not to add noise.', default=False)
parser.add_argument('--device', type=str, help='Simulator device.', default='CPU')
parser.add_argument('--random_seed', type=int, help='Seed of random numbers.', default=42424)
parser.add_argument('--traj_num', type=int, help='Number of trajectories.', default=1)
parser.add_argument('--time_interval', type=float, help='Length of time steps.', default=5)
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
evo_info['initial_state'] = tensor(basis(2,0), basis(2,0), basis(2,1))
exact_results = mesolve(Qobj(H), Qobj(psi0).unit(), times, [Qobj(op) for op in c_ops], [Qobj(op) for op in e_ops]).expect

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
markers=['', '', '', '', '']
linestyles=['-', '-', '-', '-', '-']
labels=["Site 1", "Site 2", "Site 3", "Sink", "Ground"]
plot_evo_curve(evo_data=exact_results, ax=ax_evo, times=times, 
               markers=markers, linestyles=linestyles, labels=labels, 
               xlabel='Time', ylabel='Population')

linestyles=['--', '--', '--', '--', '--']
labels=["QSD Site 1", "QSD Site 2", "QSD Site 3", "QSD Sink", "QSD Ground"]
plot_evo_curve(evo_data=np.asarray(traj_averages).T, ax=ax_evo, times=times, 
               markers=markers, linestyles=linestyles, labels=labels, 
               xlabel='Time', ylabel='Population')
plot_err_curve(evo_data=np.asarray(traj_averages).T, ref_data=np.asarray(exact_results), ax=ax_err, times=times, 
               markers=markers, linestyles=linestyles, labels=labels, 
               xlabel='Time', ylabel='Population')

ax_evo.legend()
ax_err.legend()
fig_evo.savefig(f'{file_info}_evo', dpi=600)
fig_err.savefig(f'{file_info}_err.png', dpi=600)
np.savez(f'{file_info}_data.npz', results=results)
