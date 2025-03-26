from joblib import Parallel, delayed
from qutip import *
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from qsd_quantum import single_traj_evo
from utils import *
from constants import *

parser = argparse.ArgumentParser(description='Lindblad Dynamics')

# Hamiltonian
g = 0.1
H = np.kron(PZ,PZ)-0.5*(np.kron(PX,ID) + np.kron(ID,PX))

# Lindblad operators and initial mnbd operator
op = np.sqrt(g)*PL
c_ops = [np.kron(op, ID), np.kron(ID, op)]
e_ops = [np.kron(PJ0, PJ0), np.kron(PJ1, PJ1), np.kron(PJ0, PJ1)]
psi0 = np.array([[0], [0], [0], [1]])+0j

# Hamiltonian Information
parser.add_argument('--system_name', type=str, help='Name of physical/chemical systems.', default='TFIM')
parser.add_argument('--hamiltonian', type=dict, help='System Hamiltonian.', default=H)
parser.add_argument('--channels', type=dict, help='Operators of damping channels.', default=c_ops)
parser.add_argument('--magnus_order', type=int, help='Order of Magnus expansion.', default=2)
parser.add_argument('--types', type=str, help='The type of QSD (linear or nonlinear).', default='Nonlinear')
parser.add_argument('--observation', type=list, help='The operators that need to be measured.', default=e_ops)

# Circuit Information
parser.add_argument('--param_num_per_layer', type=int, help='Name of parameters per layer.', default=7)
parser.add_argument('--layer_number', type=int, help='Number of layers.', default=3)  
parser.add_argument('--qubit_num', type=int, help='Number of qubits.', default=3)
parser.add_argument('--bit_num', type=int, help='Number of classical bits.', default=0)
parser.add_argument('--initialize', type=dict, help='The initial state.', default=np.kron(psi0, np.asarray([[1], [0]])).flatten())
parser.add_argument('--unit_str', type=str, help='The quantum circuits for each layer.', 
                    default=' RX(p{},1); RY(p{},1); RZ(p{},1); RX(p{},2); RY(p{},2); RZ(p{},2); CX(1,2); RZ(p{},2); CX(1,2);')

# Evolution Information
parser.add_argument('--method', type=str, help='Method choice of evolution, QSD or QJ.', default='QSD')
parser.add_argument('--simulator', type=str, help='Name of simulator.', default='statevector')
parser.add_argument('--device', type=str, help='Simulator device.', default='CPU')
parser.add_argument('--random_seed', type=int, help='Seed of random numbers.', default=42424)
parser.add_argument('--traj_num', type=int, help='Number of trajectories.', default=1)
parser.add_argument('--time_interval', type=float, help='Length of time steps.', default=0.25)
parser.add_argument('--step_num', type=int, help='Number of time steps.', default=101)
parser.add_argument('--parallel', type=int, help='Number of processes in Joblib.', default=1)

# Initialize
args = parser.parse_args()
hamil_info, circ_info, evo_info = initialize_info(args)
file_info = f'quantum_test/{args.system_name}/{args.traj_num}traj_{args.step_num}step_{args.types}_{args.magnus_order}order'
print_info(args)

fig_evo, ax_evo = plt.subplots()
fig_err, ax_err = plt.subplots()

# Exact Results
times = np.linspace(0, (evo_info['step_num']-1)*evo_info['time_interval'], evo_info['step_num'])
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
plot_evo_curve(evo_data=exact_results, ax=ax_evo, times=times, 
               markers=['', '', ''], linestyles=['-', '-', '-'], labels=["Exact |00>", "Exact |11>", "Exact |01>"], 
               xlabel='Time', ylabel='Population')

labels = ["QSD |00>", "QSD |11>", "QSD |01>"]
plot_evo_curve(evo_data=np.asarray(traj_averages).T, 
               ax=ax_evo, times=times, 
               markers=['', '', ''], linestyles=['--', '--', '--'], labels=labels, 
               xlabel='Time', ylabel='Population')

plot_err_curve(evo_data=np.asarray(traj_averages).T, ref_data=np.asarray(exact_results), 
               ax=ax_err, times=times, 
               markers=['', '', ''], linestyles=['-', '-', '-'], labels=labels, 
               xlabel='Time', ylabel='Population Error')

ax_evo.legend()
ax_err.legend()
fig_evo.savefig(f'{file_info}_evo.png', dpi=600)
fig_err.savefig(f'{file_info}_err.png', dpi=600)
np.savez(f'{file_info}_data.npz', results=results)
