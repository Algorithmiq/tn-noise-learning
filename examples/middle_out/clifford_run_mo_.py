import quimb
import quimb.tensor as qtn
import quimb.gen.rand as qgr

import numpy as np

import matplotlib.pyplot as plt

import stim

import cotengra as ctg
from tqdm import tqdm

import re
import json
import logging

import sys
sys.path.append('..')

from datetime import datetime
from tn_simulate_qpt import cnot_to_mpo, unitary_to_ptm
              
clifford_ops = {'I': np.array([[1, 0], [0,  1]]),
                'X': np.array([[0, 1], [1,  0]]),
                'Y': np.array([[0,-1j],[1j, 0]]),
                'Z': np.array([[1, 0], [0, -1]]),
                'C_XYZ': 0.5*np.array([[1.-1.j,-1.-1.j],[1.-1.j,1.+1.j]]),
                'C_ZYX': 0.5*np.array([[1.+1.j,1.+1.j],[-1.+1.j,1.-1.j]]),
                'H': np.sqrt(1/2)*np.array([[1.,1.],[1.,-1.]]),
                'H_XY': np.sqrt(1/2)*np.array([[0.,1.-1.j],[1.+1.j,0.]]),
                'H_YZ': np.sqrt(1/2)*np.array([[1.,-1.j],[1.j,-1.]]),
                'S': np.array([[1.,0.],[0.,1.j]]),
                'SQRT_X': 0.5*np.array([[1.+1.j,1.-1.j],[1.-1.j,1.+1.j]]),
                'SQRT_X_DAG': 0.5*np.array([[1.-1.j,1.+1.j],[1.+1.j,1.-1.j]]),
                'SQRT_Y': 0.5*np.array([[1.+1.j,-1.-1.j],[1.+1.j,1.+1.j]]),
                'SQRT_Y_DAG': 0.5*np.array([[1.-1.j,1.-1.j],[-1.+1.j,1.-1.j]]),
                'S_DAG': np.array([[1.,0.],[0.,-1.j]])}


def mpo_from_params_file(file_name: str, upper_ind_id = 'j_{}', lower_ind_id = 'i_{}', key = None, dtype=complex):
    """
    """
    print(f" > Creating MPO from parameters in file = {file_name}, specified by key = {key}!")
    with open(file_name, 'r') as file:
        params = json.load(file)
        if isinstance(key, str):
            params = params[key]
        params = {k: np.array(v, dtype = dtype) for k, v in enumerate(params.values())}

    num_qubits = len(params)
    bond_dim, phys_dim, _ = params[0].shape

    # Create MPO with correct shapes
    mpo = qtn.MPO_rand(num_qubits, bond_dim, phys_dim, upper_ind_id = upper_ind_id, lower_ind_id = lower_ind_id)
    
    # Set parameters of the MPO
    mpo.set_params(params)
    
    return mpo

def cnot_even(num_qubits, upper_ind_id = 'j_{}', lower_ind_id = 'i_{}'):
    """
    Even layer of CNOTs.
    """
    dep = qtn.MPO_identity(num_qubits, phys_dim = 4, upper_ind_id = upper_ind_id, lower_ind_id = lower_ind_id, site_tag_id = 'I{}')
    for i_pair in range(int(num_qubits / 2)):
        dep = dep.apply(cnot_to_mpo(2*i_pair, 2*i_pair+1, num_qubits), compress = False)
    return dep

def cnot_odd(num_qubits, upper_ind_id = 'j_{}', lower_ind_id = 'i_{}'):
    """
    Odd layer of CNOTs.
    """
    dep = qtn.MPO_identity(num_qubits, phys_dim = 4, upper_ind_id = upper_ind_id, lower_ind_id = lower_ind_id, site_tag_id = 'I{}')
    for i_pair in range(int((num_qubits-1) / 2)):
        dep = dep.apply(cnot_to_mpo(2*i_pair+1, 2*i_pair+2, num_qubits), compress = False)
    return dep

def mpo_dagger(mpo: qtn.MatrixProductOperator, upper_ind_id='j_{}', lower_ind_id='i_{}'):
    """
    Dagger of the MPO, since the MPO is in Pauli Transfer Matrix basis it is real-valued, hence the dagger only amounts to transposition.
    """
    num_qubits = mpo.L
    tensors_input = mpo.arrays

    tensors_data = []
    for q in range(num_qubits):
        if (q == 0) or (q == num_qubits - 1):
            tensors_data.append(tensors_input[q].transpose(0,2,1))
        else:
            tensors_data.append(tensors_input[q].transpose(0,1,3,2))

    return qtn.MatrixProductOperator(tensors_data, site_tag_id = 'I{}', upper_ind_id = upper_ind_id, lower_ind_id = lower_ind_id)

def random_clifford_circuit(num_qubits: int, depth: int):
    """
    Create a random Clifford circuit with brickwall topology of CNOTs.
    """

    # Create a dictionary with keys being the depths, and values being the sigle-qubit Clifford layers
    single_qubit_layers = {0: ['I',] * num_qubits}
    
    for d in range(1, depth + 1):
        # Sample a random Clifford gate for each qubit
        single_qubit_layers[d] = [np.random.choice(list(clifford_ops.keys())) for _ in range(num_qubits)]

    return single_qubit_layers

def unitary_layer_to_mpo(single_qubit_ops, upper_ind_id='j_{}', lower_ind_id='i_{}'):
    """
    """
    num_qubits = len(single_qubit_ops)

    tensors_data = []
    for q, op in enumerate(single_qubit_ops):    
        unitary = clifford_ops[op]
        if (q == 0) or (q == num_qubits - 1):
            tensors_data.append(unitary_to_ptm(unitary).reshape(1,4,4))
        else:
            tensors_data.append(unitary_to_ptm(unitary).reshape(1,1,4,4))

    return qtn.tensor_1d.MatrixProductOperator(tensors_data, site_tag_id = 'I{}', upper_ind_id = upper_ind_id, lower_ind_id = lower_ind_id)

def pauli_string_to_index(pauli_string: stim.PauliString):
    """
    Transform stim PauliStrings objects (e.g. '+ZYXIZ') to indices of Pauli matrices (e.g. [3,2,1,0,3]), according to the following notation:
    I -> 0, X -> 1, Y -> 2, Z -> 3
    """
    
    num_qubits = len(pauli_string)
    xs, zs = pauli_string.to_numpy()
    
    pauli_indices = []
    for q in range(num_qubits):
        if xs[q] == False:
        
            if zs[q] == False:
                pauli_indices.append(0)
            elif zs[q] == True:
                pauli_indices.append(3)
        
        elif xs[q] == True:
        
            if zs[q] == False:
                pauli_indices.append(1)
            elif zs[q] == True:
                pauli_indices.append(2)
    
    return pauli_indices

def mps_from_array(num_qubits, array_index):
    """
    """
    id = np.eye(4)
    tensors_data = []
    for q in range(num_qubits):
        array_to_insert = id[array_index[q]]
        
        if (q == 0) or (q == num_qubits -1):
            tensors_data.append(array_to_insert.reshape(1,4))
        else:
            tensors_data.append(array_to_insert.reshape(1,1,4))

    return qtn.tensor_1d.MatrixProductState(tensors_data, shape='lrp', site_ind_id='k{}')

def mpo_diagonal_number(mpo, array_index):
    """
    """
    num_qubits = mpo.L
    mps = mps_from_array(num_qubits, array_index)
    mpsH = mps.H
    mps.align_(mpo, mpsH)
    return (mpsH & mpo & mps) ^ ... 

def build_tem_map(tem_map, product_mpo, entangling_mpo, noise_inverse_mpo, bond_dim, cut_off = 1e-15, cut_off_mode = 'rel', method_='svd'):
    """
    """

    logger.info(f' > Constructing TEM map')

    product_mpo_dagger = mpo_dagger(product_mpo)
    full_unitary = entangling_mpo.apply(product_mpo, compress=False) 

    tem_map = full_unitary.apply(tem_map, compress=False)
    tem_map = tem_map.apply(product_mpo_dagger, compress=False)
    tem_map = tem_map.apply(entangling_mpo, compress=False)
    logger.info(f'  >> Bond sizes CNOTs applied: {tem_map.bond_sizes()}')

    tem_map.compress(max_bond=bond_dim, method=method_, cutoff=cut_off, cutoff_mode=cut_off_mode)
    logger.info(f'  >> Bond sizes after CNOTs compression: {tem_map.bond_sizes()}')

    tem_map = tem_map.apply(noise_inverse_mpo, compress=False)
    logger.info(f'  >> Bond sizes noise applied: {tem_map.bond_sizes()}')
            
    # Middle-Out MPO is found
    tem_map.compress(max_bond=bond_dim, method=method_, cutoff=cut_off, cutoff_mode=cut_off_mode)
    logger.info(f'  >> Bond sizes after noise compression: {tem_map.bond_sizes()}')
    logger.info(f'  >> Max bond dimension: {np.max(tem_map.bond_sizes())}') 

    return tem_map

def run_tem_clifford(num_qubits: int, single_qubit_layers: dict, pauli_obs: list, circuit_depth: int, 
                    bond_dim: int, mpo_noise: qtn.MatrixProductOperator, mpo_noise_inv: qtn.MatrixProductOperator, noise_type: str = 'spl',
                    use_previous_maps = False, verbose = False):
    """
    """

    observable = 1.0
    observable_mitigated = 1.0

    # Same noise in every layer for simplicity
    mpo_noise_layer0 = mpo_noise 
    mpo_noise_layer1 = mpo_noise
    mpo_noise_inverse_layer0 = mpo_noise_inv 
    mpo_noise_inverse_layer1 = mpo_noise_inv

    # Initiate the Pauli string
    while len(pauli_obs) < num_qubits:
        pauli_obs.insert(0, 0)

    # Input Pauli string
    pauli_string = stim.PauliString(pauli_obs) 
    logger.info(f'Initial Pauli string: {pauli_string}')

    # Initialization of MPO
    tem_mpo = qtn.MPO_identity(num_qubits, phys_dim = 4, site_tag_id = 'I{}', upper_ind_id='j_{}', lower_ind_id='i_{}')

    res_dict = {'seed': seed,
                'num_qubits': num_qubits,
                'bond_dim': bond_dim,
                'depth': circuit_depth,
                'noise_type': noise_type,
                'pauli_string': [str(pauli_string),],
                'observable': [str(observable),],
                'observable_mitigated': [str(observable),],
                'error': [0.,]
                }

    ####################################################################################
    # START OF SIMULATION: LOOPING THROUGH DIFFERENT CIRCUIT DEPTHS
    ####################################################################################
    for depth in range(1, circuit_depth + 1):
        
        logger.info(f'Depth {depth} / {circuit_depth}')
        
        if verbose:
            print("-----------------------")
            print(f"Depth {depth} / {circuit_depth} — Pauli String: {pauli_string} - O: {observable} - O mitigated: {observable_mitigated}")
    
        ################################################################################
        # NOISY SIMULATION
        ################################################################################
        # Single qubit Clifford gates
        single_qubit_ops = single_qubit_layers[depth]

        single_qubit_mpo = unitary_layer_to_mpo(single_qubit_ops)
        
        for q, op in enumerate(single_qubit_ops):
            pauli_string = pauli_string.after(stim.CircuitInstruction(op, [q]))

        print(f" > Applied layer of single unitaries — Pauli String: {pauli_string}") if verbose else None
        
        # Entangling layer (even or odd CNOTs)
        if ((depth-1) % 2) == 0:
            # Even layer of CNOTs
            for q in range(int(num_qubits / 2)):
                pauli_string = pauli_string.after(stim.CircuitInstruction("CNOT", [2*q, 2*q + 1]))
        else: 
            # Odd layer of CNOTs
            for q in range(int((num_qubits-1) / 2)):
                pauli_string = pauli_string.after(stim.CircuitInstruction("CNOT", [2*q + 1, 2*q + 2]))

        print(f" > Applied {['even', 'odd'][(depth-1) % 2]} CNOTs layer — Pauli String: {pauli_string}") if verbose else None

        # Compute noisy observable
        pauli_index = pauli_string_to_index(pauli_string)
        observable = observable * mpo_diagonal_number(mpo_noise_layer0, pauli_index)
        
        logger.info(f' > Clifford gates: {single_qubit_ops}')
        logger.info(f' > Pauli string: {pauli_string}')
        logger.info(f' > Noisy observable: {observable}')

        # Save and store results
        res_dict['pauli_string'].append(str(pauli_string))
        res_dict['observable'].append(str(observable))
        
        ################################################################################
        # MITIGATION OF OBSERVABLE VIA TENSOR NETWORK ERROR MITIGATION
        ################################################################################
        filename_tem_map = f'tem_maps/tem_map_n{num_qubits}_d{depth}_bd{bond_dim}_{noise_type}_s{seed}.quimb'
        if use_previous_maps:
            try:
                # Load Tensor Error Mitigation (TEM) map from file, if it exists
                tem_mpo = quimb.load_from_disk(filename_tem_map)
            except:
                logger.info(f"Depth {depth:2d} — cannot find a TEM map in {filename_tem_map} --> computing a new one!")
                # Build Tensor Error Mitigation (TEM) map
                if ((depth-1) % 2) == 0:
                    # Even layer of CNOTs
                    tem_mpo = build_tem_map(tem_mpo, single_qubit_mpo, cnot_even(num_qubits), mpo_noise_inverse_layer0, bond_dim)
                else: 
                    # Odd layer of CNOTs
                    tem_mpo = build_tem_map(tem_mpo, single_qubit_mpo, cnot_odd(num_qubits), mpo_noise_inverse_layer1, bond_dim)
        else:
            # Build Tensor Error Mitigation (TEM) map
            if ((depth-1) % 2) == 0:
                # Even layer of CNOTs
                tem_mpo = build_tem_map(tem_mpo, single_qubit_mpo, cnot_even(num_qubits), mpo_noise_inverse_layer0, bond_dim)
            else: 
                # Odd layer of CNOTs
                tem_mpo = build_tem_map(tem_mpo, single_qubit_mpo, cnot_odd(num_qubits), mpo_noise_inverse_layer1, bond_dim)
            
        # Save tem_map
        quimb.save_to_disk(tem_mpo, filename_tem_map)

        # Compute mitigated observable
        observable_mitigated = observable * mpo_diagonal_number(tem_mpo, pauli_index)

        logger.info(f' > Mitigated observable: {observable_mitigated}')
        if verbose: 
            print(f" > O: {observable} - O mitigated: {observable_mitigated}")
            print("-----------------------")

        res_dict['observable_mitigated'].append(str(observable_mitigated))
        res_dict['error'].append(str(np.abs(observable_mitigated) - 1.0))
        
        
    return res_dict

def main(num_qubits: int, circuit_depth, bond_dim: int, noise_data: tuple, inv_noise_key):
    """
    """

    # Select noise model and create MPO for the noise map and its inverse
    noise_type, noise_filename = noise_data
    
    mpo_noise = mpo_from_params_file(noise_filename, key = 'params_true_mpo', dtype=float)
    mpo_noise_inv = mpo_from_params_file(noise_filename, key = inv_noise_key, dtype=float) # 'params_inv_opt_mpo' or 'params_inv_true_mpo'

    print("\n")
    print("Bond size of noise MPO = ", mpo_noise.bond_sizes())
    print("Bond size of MPO^-1 = ", mpo_noise_inv.bond_sizes())
    print("\n")

    # Observable to measure (I->0, X->1, Y->2, Z->3)
    pauli_obs = [3,] * num_qubits # ZZZ...Z
    #pauli_obs = ([0,] * num_qubits).insert(3, int(num_qubits/2)) +  # ZZZ...Z
    
    single_qubit_clifford_layers = random_clifford_circuit(num_qubits, circuit_depth)

    res = run_tem_clifford(num_qubits, single_qubit_clifford_layers, pauli_obs, circuit_depth, 
                           bond_dim, mpo_noise, mpo_noise_inv, noise_type = noise_type)

    # Save results
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f'mo-clifford_nq{num_qubits}_d{circuit_depth}_bd{bond_dim}_{noise_type}_{inv_noise_key[7:]}_seed{seed}_{today}.json'
    with open(filename, 'w') as f:
        json.dump(res, f, indent = 4)

if __name__ == "__main__":

    seed = 3
    np.random.seed(seed)

    num_qubits = 10
    circuit_depth = 30
    
    bond_dim = 200

    # Import of the noise model (direct true noise, inverse of the characterised noise)
    noise_id = 'gen_dict_bigger_231127_kyiv_50q'
    noise_filename = f'cxeven_nq{num_qubits}_{noise_id}_ns10000000_global_repeated_sic_states_2023-11-30_seed108'
    noise_path = '../data/optimisation_data/' + noise_filename + '.json'

    inv_noise_key = 'params_inv_opt_mpo'

    today = datetime.today().strftime('%Y-%m-%d')
    logger = logging.getLogger(__name__)
    logging.basicConfig(handlers= [logging.FileHandler(f"logs/{noise_filename}_d{circuit_depth}_bd{bond_dim}_{inv_noise_key[7:]}_s{seed}_{today}.log", mode='w'), logging.StreamHandler()],
                        format="%(asctime)s %(message)s", 
                        level=logging.INFO)

    
    main(num_qubits, circuit_depth, bond_dim, (noise_id, noise_path), inv_noise_key = inv_noise_key)