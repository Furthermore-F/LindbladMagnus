# Variational Lindblad with Magnus Expansion

## QSD_SME_VQA

Building on the close relationship between nonlinearly unraveled Lindblad dynamics, stochastic
Magnus integrators, and variational quantum simulation, we propose a high-order scheme for solving the quantum
state diffusion equation using exponential integrators. This formulation facilitates the simulation of wavefunction
trajectories within the established framework of variational quantum algorithms for time evolution. Our
algorithm significantly enhances robustness in two key aspects: the stability of the simulation with large time
steps, and the reduction in the number of quantum trajectories required to accurately simulate the Lindblad dynamics
in terms of the ensemble average.

![Table of Contents](./plot_files/Illustration.png)

## Files:

- `classical_test`: testing the stochastic Magnus expansion-based schemes (Scheme I-IV) via the exact wavefunction trajectory method
- `quantum_test`: testing the validity of the ansatz-based variational simulation of quantum dynamics, implemented using `Qiskit`

**Three different models are simulated to examine the performance of the algorithm:**
- `test_TFIM.py`: the transverse field Ising model (TFIM) with damping
- `test_FMO.py`: the Fenna-Matthews-Olson (FMO) complex dynamics
- `test_RPM.py`: the radical pair model (RPM) for avian compass dynamics

## References:

Jia-Cheng Huang, Hao-En Li, Yi-Cheng Wang, Guang-Ze Zhang, Jun Li, Han-Shi Hu, “Towards robust variational quantum simulation of Lindblad dynamics via stochastic Magnus expansion”, [arXiv:2503.22099](https://arxiv.org/abs/2503.22099)
