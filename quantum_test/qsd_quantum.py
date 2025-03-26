import numpy as np
import re 
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit import Instruction, CircuitInstruction, Qubit, QuantumRegister
from qiskit_aer import Aer, AerSimulator, AerError
from qutip import *
import time
from timeit import timeit
from tqdm import tqdm

from utils import *
from constants import *

def measure_M_V(circ_info:dict, hamil_info:dict, simulator:AerSimulator):

    gate_num = len(circ_info['circuit'])

    initial_str = circ_info['initial_str']
    pattern = r"([^\s;]+?)\(p(\d+),(\d+)\)"
    matches = re.finditer(pattern, initial_str)
    start_index, end_index, qubit_index, gate_type = [], [], [], []
    for match in matches:
        gate_type.append(match.group(1))
        qubit_index.append(match.group(3))
        start_index.append(match.start())
        end_index.append(match.end())

    m = np.identity(circ_info['param_num']) / 8
    v = np.zeros(circ_info['param_num'])
    norm_grad = 0
    channel_expects = np.zeros(shape=hamil_info['channels_num']) + 0J

    circ_info['circuit'].save_statevector()

    ########## Construct matrix M and vector V ##########
    for i in range(circ_info['param_num']):

        gate_index_i = initial_str[:start_index[i]].count(';')+2

        circ_info['circuit'].data.insert(gate_index_i, CircuitInstruction(
            Instruction(f'c{gate_type[i][-1]}'.lower(),2,0,[]), (
                Qubit(QuantumRegister(circ_info['qubit_num'], 'q'), 0), 
                Qubit(QuantumRegister(circ_info['qubit_num'], 'q'), qubit_index[i]))
                ))
        
        # circ_info['circuit'].draw("mpl", style="iqp")
        # plt.show()

        for j in range(i+1,circ_info['param_num']):

            gate_index_j = initial_str[:start_index[j]].count(';')+3
    
            circ_info['circuit'].data.insert(gate_index_j, CircuitInstruction(
                Instruction(f'c{gate_type[j][-1]}'.lower(),2,0,[]), 
                (Qubit(QuantumRegister(circ_info['qubit_num'], 'q'), 0),
                 Qubit(QuantumRegister(circ_info['qubit_num'], 'q'), qubit_index[j]))
                ))
            circ_info['circuit'].data.insert(gate_index_j, CircuitInstruction(
                Instruction('x', 1, 0, []),
                (Qubit(QuantumRegister(circ_info['qubit_num'], 'q'), 0),)
                ))

            job_result = simulator.run(circ_info['circuit']).result().get_statevector(circ_info['circuit'])
            job_result = np.diagonal(partial_trace(job_result, list(range(circ_info['qubit_num']))[1:])) 

            del circ_info['circuit'].data[gate_index_j]
            del circ_info['circuit'].data[gate_index_j]

            m[i, j] = 0.25 * np.real(job_result[0] - job_result[1])
        
        for op, coeff in zip(hamil_info['circ_list'], hamil_info['coeff_list']):

            del circ_info['circuit'].data[gate_num:]
            circ_info['circuit'].compose(op, inplace=True)

            re_part = 0
            im_part = 0

            if np.abs(np.real(coeff)) > 1e-9:
                job_result = simulator.run(circ_info['circuit']).result().get_statevector(circ_info['circuit'])
                job_result = np.diagonal(partial_trace(job_result, list(range(circ_info['qubit_num']))[1:]))
                re_part = np.abs(job_result[0]) - np.abs(job_result[1])
            if np.abs(np.imag(coeff)) > 1e-9:
                circ_info['circuit'].data.insert(2, CircuitInstruction(
                    Instruction('s', 1, 0, []), 
                    (Qubit(QuantumRegister(circ_info['qubit_num'], 'q'), 0),)
                    ))
                
                job_result = simulator.run(circ_info['circuit']).result().get_statevector(circ_info['circuit'])
                job_result = np.diagonal(partial_trace(job_result, list(range(circ_info['qubit_num']))[1:]))
                im_part = np.abs(job_result[0]) - np.abs(job_result[1])
                del circ_info['circuit'].data[2]
            
            v[i] += np.imag(0.5J*coeff*(re_part+1J*im_part))
            
            if i == 0 and hamil_info['types'] == 'Linear':
                coeff -= coeff.conj()
                del circ_info['circuit'].data[gate_index_i]
                if np.abs(coeff) > 1e-9:
                    job_result = simulator.run(circ_info['circuit']).result().get_statevector(circ_info['circuit'])
                    job_result = np.diagonal(partial_trace(job_result, list(range(circ_info['qubit_num']))[1:]))
                    re_part = np.abs(job_result[0]) - np.abs(job_result[1])
                norm_grad += coeff * re_part
                circ_info['circuit'].data.insert(gate_index_i, CircuitInstruction(
                    Instruction(f'c{gate_type[i][-1]}'.lower(),2,0,[]), (
                        Qubit(QuantumRegister(circ_info['qubit_num'], 'q'), 0), 
                        Qubit(QuantumRegister(circ_info['qubit_num'], 'q'), qubit_index[i]))
                        ))
            
        del circ_info['circuit'].data[gate_num:-2]
        del circ_info['circuit'].data[gate_index_i]

        if i == 0 and hamil_info['types'] == 'Nonlinear':
            job_result = simulator.run(circ_info['circuit']).result().get_statevector(circ_info['circuit'])
            job_result = partial_trace(job_result, [0])
            for k in range(hamil_info['channels_num']):
                channel_expects[k] = job_result.expectation_value(hamil_info['channels'][k])

    hamil_info['channel_expects'] = channel_expects

    return m+m.T, v, norm_grad/(2J), channel_expects

def single_traj_evo(circ_info:dict, evo_info:dict, hamil_info:dict, random_seed:float):
    
    simulator = evo_info['simulator']
    hamil_info['random_seed'] = random_seed

    hamil_eff = generate_PauliOp(hamil_info=hamil_info)
    hamil_info['circ_list'], hamil_info['coeff_list'] = generate_circ(hamil_eff, circ_info['qubit_num'])
    
    results = []
    psi_norm = 1.0
    circ_info['params'] = {circ_info['theta_list'][i]: 0 for i in range(circ_info['param_num'])}

    for step_index in tqdm(range(evo_info['step_num'])):

        circ_info['circuit'] = circ_info['circuit_template'].copy()
        circ_info['circuit'] = circ_info['circuit'].assign_parameters(circ_info['params'])
        circ_info['circuit'].save_statevector()

        # Measure population
        job_result = simulator.run(circ_info['circuit']).result().get_statevector(circ_info['circuit'])
        # circ_info['circuit'].draw("mpl", style="iqp")
        # plt.show()
        # print(job_result)
        job_result = partial_trace(job_result, [0])
        job_result = np.abs(np.asarray([job_result.expectation_value(op) for op in hamil_info['observation']]))
        if hamil_info['types'] == 'Linear':
            job_result = np.abs(job_result*(psi_norm**2))
        results.append(job_result)
        print(job_result)
        del circ_info['circuit'].data[-1]

        # Runge-Kutta RK4 Method

        # The first point
        m, v, norm_grad, expect_L_1 = measure_M_V(circ_info, hamil_info, simulator)
        # m = m + 1e-5*np.identity(circ_info['param_num'])
        k_1 = np.dot(np.linalg.pinv(m), v)
        delta_norm_1 = norm_grad * psi_norm

        # The second point
        circ_info['circuit'] = circ_info['circuit_template'].copy()
        circ_info['params'] = {key: circ_info['params'][key]+evo_info['time_interval']*k_1[i]/2 
                               for i, key in enumerate(circ_info['params'])}
        circ_info['circuit'] = circ_info['circuit'].assign_parameters(circ_info['params'])
        m, v, norm_grad, expect_L_2 = measure_M_V(circ_info, hamil_info, simulator)
        # m = m + 1e-5*np.identity(circ_info['param_num'])
        k_2 = np.dot(np.linalg.pinv(m), v)
        delta_norm_2 = norm_grad * (delta_norm_1*evo_info['time_interval']/2 + psi_norm)

        # The third point
        circ_info['circuit'] = circ_info['circuit_template'].copy()
        circ_info['params'] = {key: circ_info['params'][key]+evo_info['time_interval']*(k_2[i]-k_1[i])/2 
                               for i, key in enumerate(circ_info['params'])}
        circ_info['circuit'] = circ_info['circuit'].assign_parameters(circ_info['params'])
        m, v, norm_grad, expect_L_3 = measure_M_V(circ_info, hamil_info, simulator)
        # m = m + 1e-5*np.identity(circ_info['param_num'])
        k_3 = np.dot(np.linalg.pinv(m), v)
        delta_norm_3 = norm_grad * (delta_norm_2*evo_info['time_interval']/2 + psi_norm)

        # The forth point
        circ_info['circuit'] = circ_info['circuit_template'].copy()
        circ_info['params'] = {key: circ_info['params'][key]+evo_info['time_interval']*(2*k_3[i]-k_2[i])/2 
                               for i, key in enumerate(circ_info['params'])}
        circ_info['circuit'] = circ_info['circuit'].assign_parameters(circ_info['params'])
        m, v, norm_grad, expect_L_4 = measure_M_V(circ_info, hamil_info, simulator)
        # m = m + 1e-5*np.identity(circ_info['param_num'])
        k_4 = np.dot(np.linalg.pinv(m), v)
        delta_norm_4 = norm_grad * (delta_norm_3*evo_info['time_interval'] + psi_norm)

        # Update parameters
        hamil_info['expect_L'] = 1/6*(expect_L_1+2*(expect_L_2+expect_L_3)+expect_L_4)
        circ_info['params'] = {key: circ_info['params'][key]+evo_info['time_interval']*((k_1[i]+2*k_2[i]+2*k_3[i]+k_4[i])/6-k_3[i]) 
                               for i, key in enumerate(circ_info['params'])}
        psi_norm += 1/6*evo_info['time_interval']*(delta_norm_1+2*(delta_norm_2+delta_norm_3)+delta_norm_4)

        # Update Wiener process
        hamil_info['random_seed'] = step_index+random_seed+1
        hamil_eff = generate_PauliOp(hamil_info=hamil_info)
        hamil_info['circ_list'], hamil_info['coeff_list'] = generate_circ(hamil_eff, circ_info['qubit_num'])
    
    return results
