from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice,BoundaryCondition
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import scipy


class Fermi_Hubbard():
    def __init__(self,Ms:int,U: float,J: float=1.0,BoundaryCondition:BoundaryCondition=BoundaryCondition.PERIODIC) -> None:
        self.n_sites = Ms
        self.boundary_condition = BoundaryCondition
        self.lattice = LineLattice(num_nodes=self.n_sites,boundary_condition=self.boundary_condition)
        self.fermi_hubbard_model = FermiHubbardModel(
            lattice=self.lattice.uniform_parameters(
                uniform_interaction=-J,
                uniform_onsite_potential=0.0,
            ),
            onsite_interaction=U,
        )
        self.H_fermiop = self.fermi_hubbard_model.second_q_op()
        self.Hamiltonian = JordanWignerMapper().map(self.H_fermiop)
        self.Hamiltonian_matrix = self.Hamiltonian.to_matrix()
        