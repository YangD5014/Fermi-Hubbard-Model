from mindquantum.core.operators import FermionOperator
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from mindquantum.core.operators import FermionOperator,QubitOperator
from mindquantum.core.parameterresolver import ParameterResolver
from qiskit.quantum_info import SparsePauliOp
from mindquantum.algorithm.nisq import Transform
from qiskit.quantum_info.operators import SparsePauliOp
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.algorithm.compiler import DAGCircuit
from mindquantum.simulator import Simulator



def depth_cir(cir:Circuit)->int:
    dag_cir = DAGCircuit(cir)
    return dag_cir.depth()
def convert_pauli_string(pauli_string):
    converted_string = ''
    num_qubits = len(pauli_string)
    for i in range(num_qubits):
        if pauli_string[i] == 'X':
            converted_string += f'X{num_qubits - 1 - i} '
        elif pauli_string[i] == 'Y':
            converted_string += f'Y{num_qubits - 1 - i} '
        elif pauli_string[i] == 'Z':
            converted_string += f'Z{num_qubits - 1 - i} '

    return converted_string.strip()

def convert_hamiltonian_PauliOP(qiskit_hamiltonian:SparsePauliOp)->QubitOperator:
    mindquantum_hamiltonian=[]
    for pauli_string,coeffient in qiskit_hamiltonian.to_list():
        pauli_str = convert_pauli_string(pauli_string=pauli_string)
        mindquantum_hamiltonian.append(QubitOperator(terms=pauli_str,coefficient=coeffient))
    return(sum(mindquantum_hamiltonian))


class FermiHubbard():
    def __init__(self,N_site:int=8,U:float=0.1,J:float=1.0) -> None:
        self.N_site = N_site
        self.U = U
        self.J = J
        self.n_qubit = 2*self.N_site
        self.FermiOp_Hamiltonian()
        self.mapper = JordanWignerMapper()
        self.QubitOp_Hamiltonian = self.mapper.map(self.FermiOp_Hamiltonian)
        # self.HW_hamiltonian = convert_hamiltonian_PauliOP(self.QubitOp_Hamiltonian)
        

        
        
    def FermiOp_Hamiltonian(self):
        hamiltonian = FermionicOp({}, num_spin_orbitals=2*self.N_site)
        for i in range(self.N_site - 1):
            # c_i^dagger * c_(i+1) 跳跃项 (带有自旋上和自旋下)
            hopping_term_up = FermionicOp({f"+_{i} -_{i+1}": -self.J},2*self.N_site)
            hopping_term_down = FermionicOp({f"+_{i+self.N_site} -_{i+1+self.N_site}": -self.J},2*self.N_site)
            
            # 反向跳跃项 c_(i+1)^dagger * c_i
            hopping_term_up_reverse = FermionicOp({f"+_{i+1} -_{i}": -self.J},2*self.N_site)
            hopping_term_down_reverse = FermionicOp({f"+_{i+1+self.N_site} -_{i+self.N_site}": -self.J}, num_spin_orbitals=2*self.N_site)
            
            # 将跳跃项加入哈密顿量
            hamiltonian += hopping_term_up + hopping_term_up_reverse
            hamiltonian += hopping_term_down + hopping_term_down_reverse

        # 库仑相互作用项构建：U * n_{i↑} * n_{i↓}
        for i in range(self.N_site):
            interaction_term = FermionicOp({f"+_{i} -_{i} +_{i+self.N_site} -_{i+self.N_site}": self.U}, num_spin_orbitals=self.N_site*2)
            hamiltonian += interaction_term
        self.FermiOp_Hamiltonian = hamiltonian
        
    def Hamiltonian_pool_init(self):
        self.HamiltonianPoolOp = []
        for paulistring,coeff in self.QubitOp_Hamiltonian.to_list():
            if paulistring == 'I'*self.n_qubit:
                continue
            self.HamiltonianPoolOp.append(SparsePauliOp(data=paulistring))
        # self.logger.info(f'Fermi Hubbard HamiltonianPool Init done!N-site={self.n_qubit}|length={len(self.HamiltonianPoolOp)}')

