from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz,StatePreparation
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import Hamiltonian
from qiskit.quantum_info import Statevector
from qiskit_algorithms.time_evolvers.variational.variational_principles import RealMcLachlanPrinciple
import datetime
import pickle
# 获取当前日期和时间，然后格式化为日期+小时+分钟
formatted_now = datetime.datetime.now().strftime("%m-%d %H:%M")
from qiskit.primitives.estimator import Estimator
from FermiHubbard_model import Fermi_Hubbard
from MLVP_tool import Mclachlan_distance,M,V,AVQDS_optimize_value,Energy_Distance,McLachlan_distance_optimize,AVQDS_optimize_value_new,Energy_Distance_optimize
import scipy.linalg
import numpy as np
import copy
import logging


class LMS_AVQDS_Energy():
    def __init__(self,EndTime:float,N_site:int,U:float,J: float=1.0,TimeStep: float=0.01,Threshold: float=1e-3,max_add:int=2,max_theta: float=0.005,growing_mode:int=1,dt_max:float=0.1) -> None:
        self.N_site = N_site
        self.U = U
        self.J = J
        self.Hamiltonian_initial = Fermi_Hubbard(Ms=self.N_site,U=self.U,J=self.J).Hamiltonian
        self.H = self.Hamiltonian_initial
        self.n_qubit = self.Hamiltonian_initial.num_qubits
        eignvalue,vector = np.linalg.eigh(self.Hamiltonian_initial.to_matrix())
        self.start_en=eignvalue[0]
        
        self.growing_strategy_mode = growing_mode #0- McLachlan
        
        initial_state_circuit = QuantumCircuit(self.n_qubit)
        initial_state_circuit.append(StatePreparation(Statevector(data=vector[:,0])),range(self.n_qubit))
        self.initial_state = initial_state_circuit.to_instruction(label='Initial')
    
        del initial_state_circuit
    
        self.estimator = Estimator()
        
        self.timestep = TimeStep
        self.TimeNumStep = int(EndTime/TimeStep) #总的时间步数
        self.currentTime = 0.0
        self.max_theta = max_theta
        self.endtime = EndTime
        self.logger_init()
        self.HamiltonianPool_Init()
        #self.McLachlanPrinciple = RealMcLachlanPrinciple()
        self.currentOptimalValue = []
        
        #超参数
        self.cutoff_distance = Threshold
        self.cutoff_gradient = Threshold
        self.max_iter = max_add #每一个时间步长最多加入的算符数量
        
        #记录历史
        self.OptimialValueHistory = []
        self._IndexHistory = []
        self._num_pick = 0
        self.general_step =1
        
        self.max_theta = max_theta #最大允许的θmax 论文里是 0.005
        self.max_dt = dt_max #最大允许的时间步长
        self.CircuitDepthHistory = []
        self.CircuitCNOTHistory = []
        self.AnsatzHistory = [] #存一下每一轮的Ansatz
        
        self.TimestepHistory = [] #每次的时间步长
        self.max_theta_parameter = [] #每次的最大参数
        
        self.differ_theta_unbounded=[]
        self.differ_theta_bounded=[]
        
        self.before_Matrix_History = []
        self.before_Vector_History =[]
        self.after_Matrix_History = []
        self.after_Vector_History = []
        
        self.MclachlanDistanceHistory = []
        self.MclachlanGradientHistory = []
        self.GroundStateEnergyHistory = [] #基于MLVP的能量
        self.GroundStateEnergyError = [] #基态能量误差
        self.OverlapHistory = [] #记录下瞬时精确态和瞬时变分态的重合度
        self.EnergyGradientHistory = [] #记录下瞬时能量梯度
        
        
        #记录瞬时态 瞬时能量 瞬时基态能量
        
        self.initial_ground_state = vector[:,0]  #记录H(0)的初态
        self.exact_state=[] #精确瞬时能量
        self.exact_energy=[] #精确瞬时态
        self.var_state=[]  #变分瞬时态
        self.var_energy=[] #变分瞬时能量
        self.exact_ground_state_energy=[] #精确基态能量
        
        
        
        #标志位
        self.FlagToStopAVQDS = False
        self.FlagToStopAll = False
        
        
    def Hamitonian_with_time(self,t:float):
        return LMSHamiltonian(N_site=self.N_site,gamma=gamma(t=t,T=self.T),h_z=self.hz)
    
    def logger_init(self,logger_name:str=__name__):
        
        # 定义记录器对象
        self.logger = logging.getLogger(logger_name)
        # 设置记录器级别
        self.logger.setLevel(logging.DEBUG)
        # 设置过滤器 只有被选中的可以记录
        myfilter = logging.Filter(logger_name)
        # 定义处理器-文件处理器
        filehandler = logging.FileHandler(filename=str(logger_name+formatted_now+'.log'), mode='w')
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
        self.logger.info(f'LMS HamiltonianPool Init done!N-site={self.n_qubit}')
    
    def FirstStep(self):
        self.logger.info(f'-----------------第{self.general_step}个时间步长,当前时间是{self.currentTime}-------------------')
        self.currentTime = 0.0
        self.currentAnsatz = QuantumCircuit(self.n_qubit)
        self.currentAnsatz.append(self.initial_state,range(self.n_qubit))
        self.currentIndex = 0
        self.currentTime += self.timestep
        self.logger.info(f'目前时间是{self.currentTime}')
        self.logger.info(f'起始H(0)的基态能量为={self.start_en}')

        #self.firstresult = []

        self._current_dist = 1.0
        self._current_gradient= 1.0
        self._current_iter = 0
        self._current_end_flag = False
        
        #记录 Exact 瞬时能量
        self._current_hamitonian = self.Hamitonian_with_time(t=self.currentTime)
        exp_Ht = scipy.linalg.expm(-1j*self.Hamiltonian_initial.to_matrix()*self.timestep)# U(t=0)
        
        exact_state = exp_Ht@self.initial_ground_state
        exact_state = exact_state/np.linalg.norm(exact_state)
        exact_energy = np.real(exact_state.conj().T @ self._current_hamitonian.to_matrix() @ exact_state)
        exact_energy /= (exact_state.conj().T@exact_state)
        exact_energy = np.real(exact_energy)
        
        self.logger.info(f'exact_energy={exact_energy}')
        
        self.exact_state.append(exact_state)  #保存一下 瞬时精确态 ｜exp(-iHt)|psi 0>
        self.exact_energy.append(exact_energy) #保存一下 瞬时精确能 <psi(t)|H|psi(t)>
        
        #self._current_dist,M,V = Mclachlan_distance(qc=self.currentAnsatz,initial_point=None,Hamiltonian=self._current_hamitonian)
        
        while self._current_end_flag is False: 
            self._current_iter += 1
            self.logger.info(f'本时间步长内的第{self._current_iter}次迭代')
            self._dist = []
            #self._en_dist = []
            self._gradient=[]
        
            for i in self.HamiltonianPoolOp:
                self.tmpAnsatz = copy.deepcopy(self.currentAnsatz)
                self.tmpAnsatz.append(EvolvedOperatorAnsatz(i),range(self.n_qubit))
                initial_point = np.append(self.currentOptimalValue,[0.0]*self._current_iter)
                if self.growing_strategy_mode ==0:
                    dist,M,V = Mclachlan_distance(qc=self.tmpAnsatz,initial_point=initial_point,Hamiltonian=self._current_hamitonian)
                    self._dist.append(dist)
                    self.logger.info(f'Iter={self._current_iter}|operator={i.paulis}｜Distance={self._dist[-1]}')
                    
                    #挑选出最小的 Mclachlan Distance数值
                    pick_index = np.argmin(np.abs(np.array(self._dist)))    
                    self._current_dist = np.min(np.abs(np.array(self._dist)))
                    self.MclachlanDistanceHistory.append(self._current_dist)
                    
                #energy_dist = Energy_Distance(qc=self.tmpAnsatz,initial_point=initial_point,Hamiltonian=self._current_hamitonian)
                if self.growing_strategy_mode ==1:
                    dist,M,V = Energy_Distance(qc=self.tmpAnsatz,initial_point=initial_point,Hamiltonian=self._current_hamitonian,time_diff=self.timestep)
                    self._dist.append(dist)
                    self.logger.info(f'Iter={self._current_iter}|operator={i.paulis}｜能量距离={self._dist[-1]}')
                    
                    #挑选出绝对值最小的 能量微分数值
                    pick_index = np.argmin(np.abs(np.array(self._dist)))    
                    self._current_dist = np.min(np.abs(np.array(self._dist)))
                    self.EnergyGradientHistory.append(self._current_dist)
            
            
            #更新当前的Ansatz
            self._IndexHistory.append(pick_index)
            
            self.currentAnsatz.append(EvolvedOperatorAnsatz(self.HamiltonianPoolOp[pick_index],parameter_prefix=f'Evolve{self.general_step:02d}-{self._current_iter:02d}',name=f'Operator{pick_index}-{self.general_step:03d}-{self._current_iter:02d}'),range(self.n_qubit))
            
            if self.growing_strategy_mode ==0:
                self.logger.info(f'迭代{self._current_iter}完成✅{pick_index+1} -th has been picked!Mclachlan Distance ={self._current_dist}')
            if self.growing_strategy_mode ==1:
                self.logger.info(f'迭代{self._current_iter}完成✅{pick_index+1} -th has been picked!Energy Distance ={self._current_dist}')
            #判断是否在此时间步长内结束
            if self._current_dist < self.cutoff_distance:
                # self.logger.info('达到停止条件!')
                self._current_end_flag = True
                self.logger.info(f'迭代{self._current_iter}结束✅!Mclachlan Distance ={self._current_dist}/{self.cutoff_distance}|满足停止要求😉')
                break
            else:
                self.logger.info(f'迭代{self._current_iter}继续🔄!Mclachlan Distance ={self._current_dist}/{self.cutoff_distance}')
                self._num_pick += 1
        
        #参数优化 ∇θ= M^-1@V * ∇t
        self.logger.info(f'开始参数优化,初始值={initial_point}')
        optimal_value,self.current_M,self.current_V,self.max_parameter = AVQDS_optimize_value(ansatz_unbound=self.currentAnsatz,initial_point=initial_point,Hamiltonian=self._current_hamitonian,time_diff=self.timestep)
        self.TimestepHistory.append(self.timestep)
        
        

        
        self.logger.info(f'本次的时间步长是={self.timestep}|最大变化值={self.max_parameter}')
    
        dt = min(self.max_dt,np.abs(self.max_theta/self.max_parameter))
        self.timestep = dt
        
        self.logger.info(f'时间步长已经更新={self.timestep}|{np.abs(self.max_theta/self.max_parameter)}')
        self.TimestepHistory.append(self.timestep)
        
        self.AnsatzHistory.append(self.currentAnsatz)
        self.CircuitDepthHistory.append(self.currentAnsatz.depth())#统计线路深度
        self.CircuitCNOTHistory.append(self.currentAnsatz.decompose(reps=10).count_ops()['cx'])#统计线路CNOT门个数
        self.max_theta_parameter.append(self.max_parameter)
        
        
        
        
        
        var_state = Statevector(data=self.currentAnsatz.assign_parameters(optimal_value)).data
        var_energy = var_state.conj().T @ self._current_hamitonian.to_matrix() @ var_state
        var_energy = np.real(var_energy/(var_state.conj().T@var_state))
        
        self.current_var_state = var_state/np.linalg.norm(var_state)
        self.current_var_energy = var_energy
        self.current_exact_state = exact_state/np.linalg.norm(exact_state)
        
        
        self.var_state.append(var_state) #保存一下 var_state 瞬时变分态
        self.var_energy.append(var_energy) #保存一下 var_energy 瞬时变分能量
        
        overlap = np.abs((exact_state.conj().T@var_state)**2)
        self.OverlapHistory.append(overlap)
        
        
        #e,v = np.linalg.eigh(self._current_hamitonian.to_matrix())
        
        if self.growing_strategy_mode ==0:
            self.logger.info(f'本次时间步长结束｜共选取{self._current_iter}个算符:{self._IndexHistory[-self._current_iter:]}｜最优参数为{optimal_value}｜Mclachlan Distance={self._current_dist}\n|瞬时精确能量={exact_energy}|瞬时变分能量={var_energy}|瞬时能量误差={var_energy-exact_energy}|精确与变分态重合度={overlap}')
        
        if self.growing_strategy_mode ==1:
            self.logger.info(f'本次时间步长结束｜共选取{self._current_iter}个算符:{self._IndexHistory[-self._current_iter:]}｜最优参数为{optimal_value}｜能量距离={self._current_dist}\n|瞬时精确能量={exact_energy}|瞬时变分能量={var_energy}|瞬时能量误差={var_energy-exact_energy}|精确与变分态重合度={overlap}')
            
        #print(f'optimal_value={optimal_value},type={type(optimal_value)}')
         
        self.currentTime += self.timestep
        self.currentOptimalValue=optimal_value
        self.OptimialValueHistory.append(optimal_value) 
        #self.GroundStateEnergyError.append(result.values[0]-e[0])
        
        # self.before_Matrix_History.append(self.current_M)
        # self.before_Vector_History.append(self.current_V)
        
        self.general_step += 1
        
        # print(f'M Matrix={self.current_M}\n V vector={self.current_V}')
        
    
    
        
    def NextStep(self):
        self.logger.info(f'-----------------第{self.general_step}个时间步长｜目前时间是{self.currentTime}/{self.endtime}-------------------')
        self.logger.info(f'目前时间是{self.currentTime}')

        self._current_dist = 1.0
        self._current_gradient= 1.0        
        self._current_iter = 0
        self._current_hamitonian = self.Hamitonian_with_time(t=self.currentTime)
        self._current_end_flag = False
        self._before_ansatz = copy.deepcopy(self.currentAnsatz)
        self._current_add = []
        
        
        #记录 Exact 瞬时能量
        self._current_hamitonian = self.Hamitonian_with_time(t=self.currentTime-self.timestep)
        
        exp_Ht = scipy.linalg.expm(-1j*self._current_hamitonian.to_matrix()*self.timestep)
        exact_state = exp_Ht@self.current_exact_state
        exact_state /= np.linalg.norm(exact_state)
        
        exact_energy = exact_state.conj().T @ self._current_hamitonian.to_matrix() @ exact_state
        exact_energy /= exact_state.conj().T@exact_state
        exact_energy = np.real(exact_energy)
        
        self.current_exact_state = exact_state
        self.exact_state.append(exact_state)  #保存一下 瞬时精确态 ｜exp(-iHt)|psi 0>
        self.exact_energy.append(exact_energy) #保存一下 瞬时精确能 <psi(t)|H|psi(t)>
        
        if self.growing_strategy_mode ==0:
            self._current_dist,self.current_M,self.current_V= Mclachlan_distance(qc=self._before_ansatz,initial_point=self.currentOptimalValue,Hamiltonian=self._current_hamitonian)
            self.MclachlanDistanceHistory.append(self._current_dist)
            
            
        if self.growing_strategy_mode ==1:
            self._current_dist,self.current_M,self.current_V = Energy_Distance(qc=self._before_ansatz,initial_point=self.currentOptimalValue,Hamiltonian=self._current_hamitonian,time_diff=self.timestep)
            self.EnergyGradientHistory.append(self._current_dist)
            
               
        if self._current_dist < self.cutoff_distance: #如果当前的量子态 满足停止条件
            self._current_end_flag = True
            
            if self.growing_strategy_mode ==0:
                self.logger.info(f'迭代{self._current_iter}结束✅!Mclachlan Distance ={self._current_dist}/{self.cutoff_distance}|满足停止要求😉')
                
            if self.growing_strategy_mode ==1:
                self.logger.info(f'迭代{self._current_iter}结束✅!能量距离={self._current_dist}/{self.cutoff_gradient}|满足停止要求😉')

            # self.logger.info(f'迭代{self._current_iter}结束✅!Mclachlan Distance ={self._current_dist}/{self.cutoff_distance}|满足停止要求😉|能量梯度={en_dist}')
        else:    # self.logger.info(f'使用新技术测量的Mclachlan Distance={self._current_dist2}')
            if self.growing_strategy_mode ==0:
                
                self.logger.info(f'目前Mclachlan Distance={self._current_dist}/{self.cutoff_distance}|不满足停止要求,开始寻找 Operator😉')
            if self.growing_strategy_mode ==1:
                self.logger.info(f'目前能量距离 ={self._current_dist}/{self.cutoff_gradient}|不满足停止要求,开始寻找 Operator😉')
            
            
        while self._current_end_flag is False:
            self._current_iter += 1
            if self._current_iter>self.max_iter-1:
                self._current_end_flag = True
                
            self._dist=[]
            self._en_dist = []
            self._gradient=[]
            self.logger.info(f'本时间步长内的第{self._current_iter}次迭代')


            for i in self.HamiltonianPoolOp:
                self.tmpAnsatz = copy.deepcopy(self._before_ansatz)
                #self.tmpAnsatz = self.tmpAnsatz.assign_parameters(self.currentOptimalValue)
                
                if self._current_iter > 1:
                    for j in self._current_add:
                        self.tmpAnsatz.append(j,range(self.n_qubit))
                        
                initial_point = np.append(self.currentOptimalValue,[0.0]*(self._current_iter))                
                self.tmpAnsatz.append(EvolvedOperatorAnsatz(i),range(self.n_qubit))
                
                if self.growing_strategy_mode ==0:
                    dist,_,_ = McLachlan_distance_optimize(qc=self.tmpAnsatz,initial_point=initial_point,Hamiltonian=self._current_hamitonian,before_M=self.current_M,before_V=self.current_V)
                    self._dist.append(dist)
                    self.logger.info(f'Iter={self._current_iter}|operator={i.paulis}|McLachlan Distance={dist}')
                    
                    pick_index = np.argmin(np.abs(np.array(self._dist)))    
                    self._current_dist = np.min(np.abs(np.array(self._dist)))
                    self.MclachlanDistanceHistory.append(self._current_dist)
                    
                if self.growing_strategy_mode ==1:
                    dist = Energy_Distance_optimize(qc=self.tmpAnsatz,initial_point=initial_point,Hamiltonian=self._current_hamitonian,before_M=self.current_M,before_V=self.current_V,time_diff=self.timestep)
                    self._dist.append(dist)
                    self.logger.info(f'Iter={self._current_iter}|operator={i.paulis}|能量距离={dist}')
                    
                    pick_index = np.argmin(np.abs(np.array(self._dist)))    
                    self._current_dist = np.min(np.abs(np.array(self._dist)))
                    self.EnergyGradientHistory.append(self._current_dist)
                    
                        
            #本次选中的 operator 对应的添加块
            self._current_add.append(EvolvedOperatorAnsatz(self.HamiltonianPoolOp[pick_index],parameter_prefix=f'Evolve{self.general_step:02d}-{self._current_iter:02d}',name=f'Operator{pick_index}-{self.general_step:02d}-{self._current_iter:02d}'))
            self.currentAnsatz.append(self._current_add[-1],range(self.n_qubit))
            self._IndexHistory.append(pick_index)
            

            self.logger.info(f'迭代{self._current_iter}完成✅{pick_index+1} -th has been picked!Distance ={self._current_dist}')
            
            #判断是否在此时间步长内结束
            if self._current_dist < self.cutoff_distance:
                self._current_end_flag = True
                self.logger.info(f'迭代{self._current_iter}结束✅!Distance ={self._current_dist}/{self.cutoff_distance}|满足停止要求😉')
                break
            else:
                self.logger.info(f'迭代{self._current_iter}继续🔄!Distance ={self._current_dist}/{self.cutoff_distance}|不满足停止要求😓')
                self._num_pick += 1
        
        #参数优化 ∇θ= M^-1@V * ∇t
        initial_point = np.append(self.currentOptimalValue,[0.0]*self._current_iter)
        #initial_point = [0.0]*self._current_iter
        
        self.currentAnsatz = self._before_ansatz.copy()
        for j in self._current_add:
            self.currentAnsatz.append(j,range(self.n_qubit))
            
        
        #self.logger.info(f'开始参数优化,初始值={initial_point}')
        # optimal_value,self.current_M,self.current_V = AVQDS_optimize_value(ansatz_unbound=self.currentAnsatz,initial_point=initial_point,Hamiltonian=self._current_hamitonian,time_diff=self.timestep)
        
        optimal_value,max_parameters = AVQDS_optimize_value_new(ansatz_unbound=self.currentAnsatz,initial_point=initial_point,Hamiltonian=self._current_hamitonian,time_diff=self.timestep,before_V=self.current_V,before_M=self.current_M,theta_max=self.max_theta)
        self.logger.info(f'参数优化结果如下:{optimal_value}|最大的参数={max_parameters}')
        
        dt = min(self.max_dt,np.abs(self.max_theta/max_parameters))
        self.timestep = dt
        self.logger.info(f'时间步长已经更新={self.timestep}|{np.abs(self.max_theta/max_parameters)}')  
        
        self.AnsatzHistory.append(self.currentAnsatz)
        self.CircuitDepthHistory.append(self.currentAnsatz.depth())#统计线路深度
        self.CircuitCNOTHistory.append(self.currentAnsatz.decompose(reps=10).count_ops()['cx'])#统计线路CNOT门个数   
        self.currentTime += self.timestep
        self.currentOptimalValue=optimal_value
        self.OptimialValueHistory.append(optimal_value) 
        
        # self.after_Matrix_History.append(self.current_M)
        # self.after_Vector_History.append(self.current_V)
        self.general_step += 1
        
        
        var_state = Statevector(data=self.currentAnsatz.assign_parameters(optimal_value)).data
        var_energy = var_state.conj().T @ self._current_hamitonian.to_matrix() @ var_state
        var_energy = np.real(var_energy/(var_state.conj().T@var_state))
        
        self.var_state.append(var_state) #保存一下 var_state 瞬时变分态
        self.var_energy.append(var_energy) #保存一下 var_energy 瞬时变分能量
        overlap = np.abs((exact_state.conj().T@var_state)**2)
        self.OverlapHistory.append(overlap)
            
        self.logger.info(f'本次时间步长结束｜共选取{self._current_iter}个算符:{self._IndexHistory[-self._current_iter:]}｜最优参数为{optimal_value}｜Mclachlan Distance={self._current_dist}|瞬时精确能量={exact_energy}|瞬时变分能量={var_energy}|瞬时能量误差={var_energy-exact_energy}|精确与变分态重合度={overlap}｜目前参数规模={self.currentAnsatz.num_parameters}')
        
        print(f'M Matrix={self.current_M}\nV vector={self.current_V}')
            
        
        
    def MLVP_run(self):
        self.FirstStep()
        while self.currentTime < self.endtime:
            if self.OverlapHistory[-1]<0.97:
                self.logger.info(f'模拟失败！重合度过低={self.OverlapHistory[-1]}')
                break
            self.NextStep()
            self.save()
            
    def save(self):
        with open(f'./LMS-AVQDS-Energy{formatted_now}.pkl','wb') as f:
            pickle.dump(self,f)
        
        
        
    def exact_simulation(self):
        self.currentTime = 0.0
        self.TimeHistory=[]
        self.current_exact_state = self.initial_ground_state
        while self.currentTime < self.endtime:
            self.TimeHistory.append(self.currentTime)
            self._current_hamitonian = self.Hamitonian_with_time(t=self.currentTime)
            self.currentTime += self.timestep
            exp_Ht = scipy.linalg.expm(-1j*self._current_hamitonian.to_matrix()*self.timestep)
            exact_state = exp_Ht@self.current_exact_state
            exact_state /= np.linalg.norm(exact_state)
            exact_energy = exact_state.conj().T @ self._current_hamitonian.to_matrix() @ exact_state
            
            self.current_exact_state = exact_state
            exact_energy /= exact_state.conj().T@exact_state
            exact_energy = np.real(exact_energy)
            self.exact_state.append(exact_state)  #保存一下 瞬时精确态 ｜exp
            self.exact_energy.append(exact_energy) #保存一下 瞬时精确能 <psi
        
    
    @staticmethod    
    def parameter_optimize(ansatz_unbound:QuantumCircuit,Hamiltonian:SparsePauliOp,initial_point:np.array,time_diff:float):
        M_matrix = M(qc=ansatz_unbound,initial_point=initial_point)
        V_vector = V(qc=ansatz_unbound,initial_point=initial_point,Hamiltonian=Hamiltonian)
        M_inverse = np.linalg.inv(M_matrix)
        optimal_value = M_inverse@V_vector*time_diff
        #print(f'optimal_value={optimal_value}')
        optimal_value = initial_point + optimal_value
        return optimal_value
        
        
        
        
    
                                