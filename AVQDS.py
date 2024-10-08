from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import Hamiltonian
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.time_evolvers.variational.variational_principles import RealMcLachlanPrinciple
from qiskit_algorithms.minimum_eigensolvers.adapt_vqe import AdaptVQE
from qiskit_algorithms import TimeEvolutionProblem,TimeEvolutionResult,SciPyRealEvolver
from qiskit_algorithms.time_evolvers.variational import VarQRTE
from qiskit.primitives.estimator import Estimator
from VQDS.MLVP_tool import Mclachlan_distance
from model import LMSHamiltonian,gamma
import numpy as np
import copy
import logging


class AVQDS():
    def __init__(self,Hamitonian:SparsePauliOp,EndTime:float,TimeStep: float=0.01) -> None:
        self.Hamitonian = Hamitonian
        self.mapper = JordanWignerMapper()
        self.initial_state = None
        if type(self.Hamitonian) == SparsePauliOp:
            self.H = self.Hamitonian
        else:
            self.H = self.mapper.map(self.Hamitonian.second_q_op())
            
        self.n_qubit = self.H.num_qubits
        self.timestep = TimeStep
        self.currentTime = 0.0
        self.endtime = EndTime
        self.logger_init()
        self.HamiltonianPool_Init()
        self.McLachlanPrinciple = RealMcLachlanPrinciple()
        self.currentOptimalValue = []
        
        #记录历史
        self.OptimialValueHistory = []
        self.IndexHistory = []
        
        #标志位
        self.FlagToStopAVQDS = False
        self.FlagToStopAll = False
    
    def logger_init(self,logger_name:str=__name__):
        
        # 定义记录器对象
        self.logger = logging.getLogger(logger_name)
        # 设置记录器级别
        self.logger.setLevel(logging.DEBUG)
        # 设置过滤器 只有被选中的可以记录
        myfilter = logging.Filter(logger_name)
        # 定义处理器-文件处理器
        filehandler = logging.FileHandler(filename=str(logger_name+'new'+'.log'), mode='w')
        filehandler.addFilter(myfilter)
        # formatter = logging.Formatter('%(asctime)s-%(levelname)s-\n%(message)s')
        # filehandler.setFormatter(formatter)
        # 定义处理器-控制台处理器
        concolehander = logging.StreamHandler()
        concolehander.setLevel(logging.INFO)
        # 记录器绑定handerler
        self.logger.handlers.clear()
        self.logger.addHandler(filehandler)
        self.logger.addHandler(concolehander)
        self.logger.info('logger init done!')
    
    
    def HamiltonianPool_Init(self):
        self.HamiltonianPoolOp = []
        for paulistring,coeff in self.H.to_list():
            self.HamiltonianPoolOp.append(SparsePauliOp(data=paulistring))
        self.logger.info('HamiltonianPool Init done!')
    
    def FirstStep(self):
        self.currentAnsatz = QuantumCircuit(self.n_qubit)
        self.currentIndex = 0

        self.firstresult = []
        for i in self.HamiltonianPoolOp:
            self.tmpAnsatz = copy.deepcopy(self.currentAnsatz)
            self.tmpAnsatz.append(EvolvedOperatorAnsatz(i),range(self.n_qubit))
            #self.firstresult.append(self.McLachlanPrinciple.evolution_gradient(hamiltonian=self.H,ansatz=self.tmpAnsatz,param_values=[0.0]))
        
        # pick_index = np.argmax(np.abs(np.array(self.firstresult)))    
        # max_value = np.max(np.abs(np.array(self.firstresult)))
        
        # self.IndexHistory.append(pick_index)
        # self.currentAnsatz.append(EvolvedOperatorAnsatz(self.HamiltonianPoolOp[pick_index],parameter_prefix=f'Evolve{self.currentIndex:03d}'),range(self.n_qubit))
        

        
        self.currentTime += self.timestep
        self.currentIndex += 1
        
        self.logger.info(f'First Step Done! {pick_index} -th has been picked! Optimal Value:{self.first_optimize.parameter_values[1]}')
        
        
        
        #再测一遍梯度：
        # afterresult = self.McLachlanPrinciple.evolution_gradient(hamiltonian=self.H,ansatz=self.currentAnsatz,param_values=self.currentOptimalValue)
        # pick_index = np.argmax(np.abs(np.array(self.firstresult)))    
        # max_value = np.max(np.abs(np.array(self.firstresult)))
        
        # problem = TimeEvolutionProblem(hamiltonian=self.H,time=self.timestep)
        # vqrte = VarQRTE(ansatz=self.currentAnsatz,initial_parameters=self.currentOptimalValue,
        #                 variational_principle=self.McLachlanPrinciple,estimator=Estimator(),num_timesteps=1)
        # after_optimize= vqrte.evolve(problem)
        
        
    def NextStep(self):
        
        self.tmpAnsatz = self.currentAnsatz.assign_parameters(self.currentOptimalValue[-1])
        result=[]
        for i in self.HamiltonianPoolOp:
            self.tmpAnsatz.append(EvolvedOperatorAnsatz(i),range(self.n_qubit))
            result.append(self.McLachlanPrinciple.evolution_gradient(hamiltonian=self.H,ansatz=self.tmpAnsatz,param_values=[0.0]))
            
            
        
        
    def GeneralProcess(self):
        start_time = self.currentTime
        self.logger.info(f'目前时间是{self.currentTime}')
        if self.FlagToStopAVQDS:
            self.logger.info(f'这次时间步长中的AVQDS已经结束!从{start_time}到{self.currentTime}')
            return
        
        
        
        
        
        
        
        
    
                                