import numpy as np
from qiskit.circuit import QuantumCircuit, parameter
from qiskit.quantum_info import Statevector,SparsePauliOp
import copy
import time
from functools import wraps

'''
本文件是用来计算 Mclachlan_distance 相关的功能函数
'''
def timeit(func):
    @wraps(func)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return timed


def quantum_state(pqc:QuantumCircuit,parameter:np.array)->np.array:
    # pqc = pqc.bind_parameters({param:val for param,val in zip(pqc.parameters,parameter)})
    #pqc = pqc.assign_parameters(dict(zip(pqc.parameters.data,parameter)))
    pqc = pqc.assign_parameters(parameters=parameter)
    state = Statevector(data=pqc)
    return state.data

def quantum_state_dagger(pqc: QuantumCircuit, parameter: np.array) -> np.array:
    state_vector = quantum_state(pqc, parameter)
    # 对量子态向量进行共轭转置操作
    state_dagger = state_vector.conjugate().transpose()
    return state_dagger

# 定义数值微分函数
def numerical_derivative(func,pqc: QuantumCircuit, parameter:np.array,index:int, delta=1e-3):
    parameter_copy = copy.deepcopy(parameter)
    parameter_copy[index] = parameter[index] + delta
    forward = func(pqc,parameter_copy)
    
    parameter_copy = copy.deepcopy(parameter)
    parameter_copy[index] = parameter[index] - delta
    backward = func(pqc,parameter_copy)
    return (forward - backward) / (2 * delta)


def QFI_last_term(qc: QuantumCircuit,initial_point:np.array):
    M = np.zeros((qc.num_parameters,qc.num_parameters))
    for index_i in range(0,qc.num_parameters):
        gradient_left_i = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_i)
        #term1 = np.dot(gradient_left_i,quantum_state(qc,initial_point))
        
        term1 = gradient_left_i@quantum_state(qc,initial_point)
        for index_j in range(0,qc.num_parameters):
            gradient_right_j = numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_j)
            #term2 = np.dot(quantum_state(qc,initial_point),gradient_right_j)
            term2 = quantum_state_dagger(qc,initial_point)@gradient_right_j
            M[index_i,index_j] = 4 * np.real(term1 * term2)
            
    return M
    
    
def QFI_first_term(qc: QuantumCircuit,initial_point: np.array)->np.array:
    M = np.zeros((qc.num_parameters,qc.num_parameters))
    for index_i in range(0,qc.num_parameters):
        for index_j in range(0,qc.num_parameters):
            left = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_i)
            right = numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_j)
            M[index_i,index_j]= 4* np.real(left@right)
    return M


def QFI(qc: QuantumCircuit,initial_point: np.array)->np.array:
    M1 = QFI_first_term(qc,initial_point)
    M2 = QFI_last_term(qc,initial_point)
    print('new')
    return M1 - M2
        

def M(qc: QuantumCircuit,initial_point: np.array)->np.array:
    '''
        返回的是一个矩阵 M
    '''
    M = np.zeros((qc.num_parameters,qc.num_parameters))
    quantum_right = quantum_state(pqc=qc,parameter=initial_point)
    #print(f'quantum_right={quantum_right}')
    for index_i in range(0,qc.num_parameters):
        
        
        d_i_left = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_i)
        
        for index_j in range(index_i,qc.num_parameters):

            d_j_left = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_j)
            d_j_right = numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_j)

            term1 = d_i_left@d_j_right
            term2 = (d_i_left@quantum_right) * (d_j_left@quantum_right)
            M[index_i,index_j] = 2*np.real(term1 + term2)
            M[index_j,index_i] = 2*np.real(term1 + term2)
            
    return M
            
            
            
# def QFI_My(qc: QuantumCircuit,initial_point: np.array)->np.array:
#     '''
#         返回的是一个矩阵 M
#     '''
#     M = np.zeros((qc.num_parameters,qc.num_parameters))
#     quantum_right = quantum_state(pqc=qc,parameter=initial_point)
#     #print(f'quantum_right={quantum_right}')
#     for index_i in range(0,qc.num_parameters):
        
        
#         d_i_left = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_i)
        
#         for index_j in range(0,qc.num_parameters):

#             d_j_left = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_j)
#             d_j_right = numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_j)

#             term1 = d_i_left@d_j_right
#             term2 = (d_i_left@quantum_right) * (d_j_left@quantum_right)
#             M[index_i,index_j] = 2*np.real(term1 - term2)
#     return M

def M_optimize(qc: QuantumCircuit,initial_point: np.array,before_M:np.array)->np.array:
    '''
    根据论文中的方法来节约计算时间 M矩阵的计算只需要更新最后一行 和最后一列(对称过去)
    因为首先在左上角载入之前的矩阵 before_M 然后只需要计算最后一行和最后一列
    '''
    before_M_size = before_M.shape[0]
    M_matrix = np.zeros((len(initial_point),len(initial_point)))
    if M_matrix.shape[0] == before_M_size:
        return before_M
    else:
        M_matrix[0:before_M_size,0:before_M_size] = before_M 
              
    new_parameters = initial_point[before_M_size:]
    quantum_right = quantum_state(pqc=qc,parameter=initial_point)
    quantum_right = quantum_state(pqc=qc,parameter=initial_point)
    #quantum_left = quantum_right.conj().T
        
    for index_i in range(before_M_size,before_M_size+len(new_parameters)):
        for index_j in range(len(initial_point)):
            term1_left = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_i)
            term1_right = numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_j)
            term1 = 2*np.real(term1_left@term1_right)
            
            term2 = term1_left@quantum_right*(numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_j)@quantum_right)
            term2 = 2*np.real(term2)
            M_matrix[index_i,index_j] = term1 + term2
            M_matrix[index_j,index_i] = term1 + term2
            
    return M_matrix


def V_optimize(qc: QuantumCircuit,initial_point: np.array,before_V:np.array,Hamiltonian: SparsePauliOp)->np.array:
    before_V_size = before_V.shape[0]
    V_vector = np.zeros((len(initial_point)))
    if before_V_size == len(initial_point):
        return before_V
    else:
        V_vector[0:before_V_size] = before_V
        new_parameters = initial_point[before_V_size:]
    
    quantum_right = quantum_state(pqc=qc,parameter=initial_point)
    quantum_left = quantum_right.conj().T
    
    for index_i in range(before_V_size,before_V_size+len(new_parameters)):
        term1 = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_i)@Hamiltonian.to_matrix()@quantum_right
        term2 = quantum_left@numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_i)*np.real(quantum_left@Hamiltonian.to_matrix()@quantum_right)
        
        V_vector[index_i] = 2*np.imag(term1+term2)
    
    return V_vector

@timeit
def McLachlan_distance_optimize(qc: QuantumCircuit,initial_point: np.array,Hamiltonian: SparsePauliOp,before_M: np.array,before_V:np.array):
    '''
    尝试使用计算优化后的McLachlan_distance 计算程序
    '''
    H = Hamiltonian.to_matrix()
    H2 = H@H
    quantum_state_right = quantum_state(pqc=qc,parameter=initial_point)
    quantum_state_left = quantum_state_dagger(pqc=qc,parameter=initial_point)
    
    variance_H = (quantum_state_left@H2@quantum_state_right) - (quantum_state_left@H@quantum_state_right)**2        
    variance_H = 2*np.real(variance_H)
    M_matrix = M_optimize(qc=qc,initial_point=initial_point,before_M=before_M)
    M_inverse = get_minv(a=M_matrix)
    #M_inverse = np.linalg.pinv(M_matrix)

    V_vector = V_optimize(qc=qc,initial_point=initial_point,Hamiltonian=Hamiltonian,before_V=before_V)

    # term_sum =0.0
    # for index_i in range(0,qc.num_parameters):
    #     for index_j in range(0,qc.num_parameters):
    #         m_inv_ij = M_inverse[index_i,index_j]
    #         v_i = V_vector[index_i]
    #         v_j = V_vector[index_j]
    #         term_sum += v_i*m_inv_ij*v_j
    term_sum = V_vector.conj().T@M_inverse@V_vector
                
    # print(f'sum = {term_sum}')
    L2 = variance_H - term_sum
    return L2,M_matrix,V_vector
    
    
    
    
    
    
    
    
    
def V(qc: QuantumCircuit,initial_point: np.array,Hamiltonian: SparsePauliOp)->np.array:
    v = np.zeros((qc.num_parameters))
    quantum_state_right = quantum_state(pqc=qc,parameter=initial_point)
    quantum_state_left = quantum_state_dagger(pqc=qc,parameter=initial_point)
    
    for index_i in range(0,qc.num_parameters):
        d_i_left = numerical_derivative(func=quantum_state_dagger,pqc=qc,parameter=initial_point,index=index_i)
        d_i_right = numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_i)
        
        term1 = d_i_left@Hamiltonian.to_matrix()@quantum_state_right
        term2 = (quantum_state_left@d_i_right)*np.real(quantum_state_left@Hamiltonian.to_matrix()@quantum_state_right)
        
        # term1 = np.vdot(d_i_left@Hamiltonian.to_matrix(),quantum_state_right)
        # term2 = np.vdot(quantum_state_left,d_i_right)*np.vdot(quantum_state_left,Hamiltonian.to_matrix()@quantum_state_right)
        v[index_i] = 2*np.imag(term1 + term2)
        
    return v
    
    
    
    
  
def Mclachlan_distance(qc: QuantumCircuit,initial_point: np.array,Hamiltonian: SparsePauliOp):
    '''
    计算 Mclachlan distance 返回值是一个数值
    '''
    H = Hamiltonian.to_matrix()
    H2 = H@H
    quantum_state_right = quantum_state(pqc=qc,parameter=initial_point)
    quantum_state_left = quantum_state_right.conjugate().transpose()
    #print(f'<H**2>{(quantum_state_left@H2@quantum_state_right)}')
    
    variance_H = (quantum_state_left@H@H@quantum_state_right) - (quantum_state_left@H@quantum_state_right)**2
    variance_H = 2*np.real(variance_H)
    
    M_matrix = M(qc=qc,initial_point=initial_point)
    
    M_inverse = get_minv(a=M_matrix)
    #M_inverse = np.linalg.pinv(M_matrix)
    
    
    V_vector = V(qc=qc,initial_point=initial_point,Hamiltonian=Hamiltonian)
    #print(f'V_vector={V_vector},M={M_matrix}')
    #print(f"Origin|M_matrix={M_matrix}\nV_vector={V_vector}")
    
    term_sum = V_vector.conj().T@M_inverse@V_vector
    
    L2 = variance_H - term_sum
    return L2,M_matrix,V_vector
            

def Energy_Distance(qc: QuantumCircuit,initial_point: np.array,Hamiltonian: SparsePauliOp,time_diff:float):
    '''
    计算能量的时间倒数 作为新的 growing strategy
    '''
    #quantum_state_right = quantum_state(pqc=qc,parameter=initial_point)
    print(f'{__name__}的参数是:{initial_point}')
    quantum_state_left = quantum_state_dagger(pqc=qc,parameter=initial_point)
    H = Hamiltonian.to_matrix()
    M_matrix = M(qc=qc,initial_point=initial_point)
    V_vector = V(qc=qc,initial_point=initial_point,Hamiltonian=Hamiltonian)
    M_inverse = get_minv(a=M_matrix)
    
    theta = M_inverse@V_vector
    #M_inverse = np.linalg.pinv(M_matrix)# M的逆矩阵
    
    result=0.0
    #term1_sum = 0.0
    for index_i in range(0,qc.num_parameters):
        term1 = np.real(quantum_state_left@H@numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_i))# Re(<psi|H|d psi/d theta_u>)
        result += term1*theta[index_i]
        
    
        # for index_j in range(0,qc.num_parameters):
        #     M_inverse_uv = M_inverse[index_i,index_j] #M^(-1)_{uv}
        #     V_vector = V(qc=qc,initial_point=initial_point,Hamiltonian=Hamiltonian) #V_v
        #     result+=term1*M_inverse_uv*V_vector[index_j]# ∑_uv Re(<psi|H|d psi/d theta_u>)*M^-1*V
    result = result*time_diff
    return result,M_matrix,V_vector
            
def Energy_Distance_optimize(qc: QuantumCircuit,initial_point: np.array,Hamiltonian: SparsePauliOp,before_M: np.array,before_V:np.array,time_diff:float):
    '''
    计算能量差 作为新的 growing strategy
    '''
    print(f'{__name__}的参数是:{initial_point}')
    quantum_state_right = quantum_state(pqc=qc,parameter=initial_point)
    quantum_state_left = quantum_state_dagger(pqc=qc,parameter=initial_point)
    H = Hamiltonian.to_matrix()
    M_matrix = M_optimize(qc=qc,initial_point=initial_point,before_M=before_M)
    V_vector = V_optimize(qc=qc,initial_point=initial_point,Hamiltonian=Hamiltonian,before_V=before_V)
    #M_inverse = np.linalg.pinv(M_matrix)# M的逆矩阵
    M_inverse = get_minv(a=M_matrix)
    theta = M_inverse@V_vector
    
    result = 0.0
    for index_i in range(0,qc.num_parameters):
        term1 = np.real(quantum_state_left@H@numerical_derivative(func=quantum_state,pqc=qc,parameter=initial_point,index=index_i))# Re(<psi|H|d psi/d theta_u>)
        result += term1*theta[index_i]
    
    result = result*time_diff
    return result

    
def AVQDS_optimize_value(ansatz_unbound:QuantumCircuit,Hamiltonian:SparsePauliOp,initial_point:np.array,time_diff:float):
    '''
    基于 AVQDS 算法进行参数优化 dθ = M^(-1)V*dt 
    '''
    M_matrix = M(qc=ansatz_unbound,initial_point=initial_point)
    V_vector = V(qc=ansatz_unbound,initial_point=initial_point,Hamiltonian=Hamiltonian)
    M_inverse = get_minv(a=M_matrix)
    
    M_inv_V = M_inverse@V_vector
    M_inv_V = np.clip(M_inv_V,a_min=-10,a_max=10) #和论文里一样限制了一下 哦莫
    
    #M_inverse = np.linalg.pinv(M_matrix)
    optimal_value = M_inverse@V_vector*time_diff
    max_parameters = max(np.abs(M_inverse@V_vector))
    
    optimal_value = initial_point + optimal_value
    
    return optimal_value,M_matrix,V_vector,max_parameters


def AVQDS_optimize_value_new(ansatz_unbound:QuantumCircuit,Hamiltonian:SparsePauliOp,initial_point:np.array,time_diff:float,before_M: np.array,before_V:np.array,theta_max:float=20):
    '''
    基于 AVQDS 算法进行参数优化 dθ = M^(-1)V*dt 
    是否需要提供 dt?? ｜目前尝试不提供 dt dt用自己算出来的
    '''
    M_matrix = M_optimize(qc=ansatz_unbound,initial_point=initial_point,before_M=before_M)
    V_vector = V_optimize(qc=ansatz_unbound,initial_point=initial_point,Hamiltonian=Hamiltonian,before_V=before_V)
    M_inverse = get_minv(a=M_matrix)
    M_inv_V =M_inverse@V_vector
    M_inv_V = np.clip(M_inv_V,a_min=-10,a_max=10)
    
   
    max_parameters = max(np.abs(M_inverse@V_vector))
    
    
    optimal_value = M_inv_V*time_diff    
    # for index,i in enumerate(optimal_value):
    #     if i>theta_max:
    #         i=theta_max
    #         optimal_value[index] = i
    #     elif i<-1*theta_max:
    #         i=-1*theta_max
    #         optimal_value[index] = i
    #print(f'optimal_value={optimal_value}')
    optimal_value = initial_point + optimal_value
    return optimal_value,max_parameters

def get_minv(a, delta=2*1e-6):
    ap = a + delta*np.eye(a.shape[0])
    ainv = np.linalg.pinv(ap)
    return ainv