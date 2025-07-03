from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit import CircuitInstruction, Instruction, Qubit
from qiskit.circuit import library
from qiskit import QuantumRegister, transpile
from qiskit_aer import Aer, AerSimulator, AerError, noise
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, fake_provider
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info import commutator, Statevector, Operator
from argparse import Namespace
import re
import matplotlib.pyplot as plt
import numpy as np
import mthree

from constants import *

def parse_gate(circuit:QuantumCircuit, command:str, param_vector:ParameterVector):
    
    if re.match(r"H\((.*)\)", command):
        qubit = int(command[2])
        circuit.h(qubit)
    elif re.match(r"S\((.*)\)", command):
        qubit = int(command[2])
        circuit.s(qubit)
    elif re.match(r"X\((.*)\)", command):
        qubit = int(command[2])
        circuit.x(qubit)
    elif re.match(r"Y\((.*)\)", command):
        qubit = int(command[2])
        circuit.y(qubit)
    elif re.match(r"Z\((.*)\)", command):
        qubit = int(command[2])
        circuit.z(qubit)
    elif re.match(r"RX\(p(.*),(.*)\)", command):
        params = re.findall(r"RX\(p(.*),(.*)\)", command)[0]
        circuit.rx(param_vector[int(params[0])], int(params[1]))
    elif re.match(r"RY\(p(.*),(.*)\)", command):
        params = re.findall(r"RY\(p(.*),(.*)\)", command)[0]
        circuit.ry(param_vector[int(params[0])], int(params[1]))
    elif re.match(r"RZ\(p(.*),(.*)\)", command):
        params = re.findall(r"RZ\(p(.*),(.*)\)", command)[0]
        circuit.rz(param_vector[int(params[0])], int(params[1]))
    elif re.match(r"CX\((.*),(.*)\)", command):
        params = re.findall(r"CX\((.*),(.*)\)", command)[0]
        circuit.cx(int(params[0]), int(params[1]))
    elif re.match(r"CY\((.*),(.*)\)", command):
        params = re.findall(r"CY\((.*),(.*)\)", command)[0]
        circuit.cy(int(params[0]), int(params[1]))
    elif re.match(r"CZ\((.*),(.*)\)", command):
        params = re.findall(r"CZ\((.*),(.*)\)", command)[0]
        circuit.cz(int(params[0]), int(params[1]))
    elif re.match(r"CI\((.*),(.*)\)", command):
        None
    elif re.match(r"CCX\((.*),(.*),(.*)\)", command):
        params = re.findall(r"CCX\((.*),(.*),(.*)\)", command)[0]
        circuit.ccx(int(params[0]), int(params[1]), int(params[2]))
    elif re.match(r"M\((.*)\)", command):
        qubit = int(re.findall(r"M\((.*)\)", command)[0])
        circuit.measure(qubit, qubit)
    else:
        raise ValueError(f"Not Found: {command}")

def create_circuit_from_string(circ_info:dict):
    
    circuit = QuantumCircuit(circ_info['qubit_num'], circ_info['bit_num'])
    if circ_info['initialize']:
        circuit.initialize(circ_info['initialize'])
    commands = circ_info['create_str'].split(';')
    for command in commands:
        parse_gate(circuit, command.strip(), circ_info['theta_list'])
    return circuit

def initialize_info(args:Namespace) -> tuple[dict, dict, dict]:

    print('############ Parameter Initialization ############')
    hamil_info, circ_info, evo_info = {}, {}, {}

    hamil_info['system'] = SparsePauliOp.from_operator(Operator(args.hamiltonian))
    hamil_info['channels'] = [SparsePauliOp.from_operator(Operator(op)) for op in args.channels]
    hamil_info['order'] = args.magnus_order
    hamil_info['channels_num'] = len(args.channels)
    hamil_info['qubit_num'] = args.qubit_num-1
    hamil_info['observation'] = [SparsePauliOp.from_operator(Operator(op)) for op in args.observation]
    hamil_info['types'] = args.types
    hamil_info['time_interval'] = args.time_interval

    circ_info['qubit_num'] = args.qubit_num
    circ_info['bit_num'] = args.bit_num
    circ_info['initialize'] = Statevector(args.initialize/np.linalg.norm(args.initialize))
    circ_info['param_num'] = args.param_num_per_layer*args.layer_number
    circ_info['theta_list'] = ParameterVector('θ', length=circ_info['param_num'])
    circ_info['theta_index'] = np.arange(circ_info['param_num'])
    initial_str = 'H(0);' + args.unit_str*args.layer_number + ' H(0)'
    circ_info['create_str'] = circ_info['initial_str'] = initial_str.format(*circ_info['theta_index'])
    circ_info['circuit'] = create_circuit_from_string(circ_info=circ_info)
    circ_info['circuit_template'] = create_circuit_from_string(circ_info=circ_info)
    circ_info['circuit'] = transpile(circ_info['circuit'], optimization_level=1)

    evo_info['method'] = args.method
    backend = fake_provider.FakeBelemV2()
    evo_info['simulator_ideal'] = AerSimulator(method=args.simulator)
    evo_info['simulator_noise'] = backend
    evo_info['mitigation'] = mthree.M3Mitigation(backend)
    evo_info['traj_num'] = args.traj_num
    evo_info['time_interval'] = args.time_interval
    evo_info['step_num'] = args.step_num
    evo_info['noisy_simulation'] = args.noisy_simulation

    channel_expects = np.zeros(shape=hamil_info['channels_num']) + 0J
    if hamil_info['types'] == 'Nonlinear':
        for k in range(hamil_info['channels_num']):
            channel_expects[k] = circ_info['initialize'].expectation_value(LM, qargs=[1])
    hamil_info['channel_expects'] = channel_expects

    return hamil_info, circ_info, evo_info

def generate_PauliOp(hamil_info:dict) -> SparsePauliOp:
    """Construct Pauli strings from Hamiltonian information.
    
    Args:
        args (Namespace): Hamiltonian information.

    Returns:
        hamil_info (dict):
            'system': dict, Pauli strings and its coefficients.
            'op_in_circ': dict, Pauli strings and corresponding quantum circuits.
            'channels': dict, operator of damping channels and coefficients.
            'time': (optional) float, time-dependent Hamiltonian
    """

    hamil_eff = hamil_info['system']
    np.random.seed(hamil_info['random_seed'])

    p = 1000 # Approximation order
    k = hamil_info['channels_num'] # Wiener increments number
    dt = hamil_info['time_interval'] # Time interval
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

    X_0 = -1J*hamil_eff+sum([-0.5*(op.adjoint()+op)@op+2*np.real(e_op)*op 
                             for op, e_op in zip(hamil_info['channels'], hamil_info['channel_expects'])])
    hamil_eff = (X_0*dt).copy()
    if hamil_info['order'] > 0:
        for i in range(k):
            hamil_eff += hamil_info['channels'][i]*np.sqrt(dt)*xis[i]
            if hamil_info['order'] > 1:
                _comm = commutator(X_0, hamil_info['channels'][i])
                hamil_eff += _comm*a0[i]*dt/2
                for j in range(i+1,k):
                    _comm_ij = commutator(hamil_info['channels'][i], hamil_info['channels'][j])
                    if np.sum(np.abs(_comm_ij)) != 0:
                        hamil_eff += 0.5*_comm_ij*((a0[j]*xis[i]-a0[i]*xis[j])*np.sqrt(dt)+2*dt*aij[j,i])
                if hamil_info['order'] > 2:
                    _comm = commutator(X_0, _comm)
                    hamil_eff += _comm*(dt**2)*(np.sqrt(dt*alpha_p)*phis[i]+np.sqrt(dt/2)*np.sum([etas[i,l]/(l+1)**2 for l in range(p)])/np.pi)/(2*np.pi)
                    if hamil_info['order'] > 3:
                        _comm = commutator(X_0, _comm)
                        hamil_eff += _comm*dt*dt*dt*c0[i]
    else:
        for i in range(k):
            hamil_eff += hamil_info['channels'][i]*np.sqrt(dt)*xis[i]
        hamil_eff -= dt*sum([-0.5*op@op+2*np.real(e_op)*op for op, e_op in zip(hamil_info['channels'], hamil_info['channel_expects'])]) # Euler-Maruyama

    return ((hamil_eff)/(-1J*dt)).simplify().sort()

def generate_circ(hamil_eff:SparsePauliOp, qubit_num:int) -> list:

    circ_list = []
    coeff_list = []
    for ops in hamil_eff:
        new_gate = QuantumCircuit(qubit_num, 0)
        for k, op in enumerate(reversed(str(ops.paulis[0]))):
            if op != 'I':
                new_gate.append(CircuitInstruction(
                    Instruction(f'c{op}'.lower(),2,0,[]),
                    (Qubit(QuantumRegister(qubit_num, 'q'), 0),
                     Qubit(QuantumRegister(qubit_num, 'q'), k+1)),
                     ))
        new_gate.h(0)
        new_gate.save_statevector()
        circ_list.append(new_gate)
        coeff_list.append(ops.coeffs[0])

    return circ_list, coeff_list

def plot_circ(circ:QuantumCircuit, save_name:str) -> None:

    circ.draw("mpl", style="iqp", 
              filename=save_name, 
              scale=3)
    
    return None

def print_info(args:Namespace) -> None:

    print(f'The system name is {args.system_name}.')
    print(f"Quantum circuit simulator backend: {AerSimulator(method=args.simulator)}.")
    print(f'The unravel method is {args.method}, the type is {args.types} and the order of Magnus is {args.magnus_order}.')
    print(f'There are {args.param_num_per_layer*args.layer_number} parameters and {args.qubit_num} qubits in circuit.')
    print(f'There are {args.traj_num} trajectories and {args.step_num} steps, dt = {args.time_interval}.')

    return None

def plot_evo_curve(evo_data:np.ndarray, 
                   ax:plt.axes,
                   times:np.ndarray, 
                   markers:list,
                   linestyles:list,
                   labels:list,
                   xlabel:str,
                   ylabel:str
                   ) -> None:
    
    if not len(evo_data) == len(markers) == len(linestyles) == len(labels):
        print('The input parameters have different lengths!')
        raise SystemExit
    else:
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for i, data in enumerate(evo_data):
            ax.plot(times, data, marker=markers[i], linestyle=linestyles[i], label=labels[i])

    return

def plot_err_curve(evo_data:np.ndarray, 
                   ref_data:np.ndarray,
                   ax:plt.axes,
                   times:np.ndarray, 
                   markers:list,
                   linestyles:list,
                   labels:list,
                   xlabel:str,
                   ylabel:str
                   ) -> None:
    if not len(evo_data) == len(markers) == len(linestyles) == len(labels):
        print('The input parameters have different lengths!')
        raise SystemExit
    else:
        evo_data -= ref_data
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for i, data in enumerate(evo_data):
            ax.plot(times, np.abs(data), marker=markers[i], linestyle=linestyles[i], label=labels[i])

    return