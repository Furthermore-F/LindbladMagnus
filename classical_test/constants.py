import numpy as np
from qutip import *
from qiskit.quantum_info.operators import SparsePauliOp

# Pauli Matrices
PX = np.array([[0,1], [1,0]])+0j
PY = np.array([[0,-1J], [1J,0]])
PZ = np.array([[1,0], [0,-1]])+0j
PS = np.array([PX, PY, PZ])
ID = np.array([[1,0], [0,1]])+0j

PR = np.array([[0,0], [1,0]])+0j
PL = np.array([[0,1], [0,0]])+0j

# Projection Matrices
PJ0 = np.array([[1,0], [0,0]])+0j
PJ1 = np.array([[0,0], [0,1]])+0j
PJA = Qobj(np.array([[1,0], [0,0]])+0j)
PJB = Qobj(np.array([[0,0], [0,1]])+0j)

LM = SparsePauliOp(["X"], [0.5]) + SparsePauliOp(["Y"], [0.5J])
LP = SparsePauliOp(["X"], [0.5]) - SparsePauliOp(["Y"], [0.5J])

