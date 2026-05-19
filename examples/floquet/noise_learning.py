import sys
sys.path.append('..')

# Qiskit imports
import qiskit as qk

# General imports
import json
import numpy as np

# JAX-like import
import jax
import optax
from jax import config
config.update("jax_enable_x64", True)

# Custom imports
import noise_channels
from lpdo_utils import build_lpdo, lpdo_to_mpo
import lpdo_utils
from training_functions import optimize_tn, optimize_trace_preservation
import dataset_utils

import quimb as qu
qu.core._NUM_THREAD_WORKERS = 3

#import numba
#numba.set_active_threads(3)

seed = 34343400
rnd_key = jax.random.PRNGKey(seed)
np.random.seed(seed)

######################################
# IMPORT EXPERIMENTAL TRAINING DATA
######################################
data_file = './stefanos_data_from_2024-07-22_odd'
with open(data_file + '.json') as f:
    exp_data = json.load(f)

#qasm_qc, noise_str = exp_data['qc'], exp_data['noise']
qasm_qc = exp_data['qc']

experiments = {'states': exp_data['states'], 'povms': exp_data['povms'], 'counts': exp_data['counts']}
#print("Noise model:", noise_str)

# Ideal unitary / quantum circuit (noise-free)
qc = qk.QuantumCircuit.from_qasm_str(qasm_qc)
num_qubits = qc.num_qubits
print("Ideal quantum circuit:")
print(qc)


######################################
# BUILD GROUND-TRUTH NOISE LPDO
######################################
#true_lpdo = noise_channels.select_noise_channel(noise_str, tensor_type = 'lpdo', num_qubits = num_qubits)
#true_mpo = noise_channels.select_noise_channel(noise_str, tensor_type = 'mpo', num_qubits = num_qubits)
#print("||MPO[simulation] - MPO[LPDO]||:", lpdo_to_mpo(true_lpdo).distance(true_mpo))


########################################################
# INITIAL PARAMETERS AND TENSOR NETWORK STRUCTURE
########################################################
virtual_bd = 4
kraus_bd = 4
init_lpdo = build_lpdo(num_qubits=num_qubits, 
                       virtual_bd=virtual_bd, 
                       kraus_bd=kraus_bd, 
                       init_method='normal_complex')

# Bring initial LPDO close to a physical channel by making it trace preserving
init_lpdo, tnopt = optimize_trace_preservation(init_lpdo, 
                                               optimizer='L-BFGS-B', 
                                               steps=2500, 
                                               return_lpdo=True, 
                                               plot_results=False)

print(f"Tr[LPDO] = {lpdo_utils.trace_lpdo(init_lpdo)} (={2**num_qubits}?)")


########################################################
# PROCESS DATASET
########################################################
batch_size = 250
build_marginals = None

counts = exp_data['counts']
states = exp_data['states']
povms = exp_data['povms']

batched_hyper_tns, batched_params_tns, batched_skeletons_tns = dataset_utils.create_training_dataset(counts, states, povms, 
                                                                                                     batch_size=batch_size, 
                                                                                                     shuffle=True, 
                                                                                                     build_marginals=build_marginals, 
                                                                                                     evolve_with_unitary=qc)

processed_experiments_dict = {'batched_hyper_tns': batched_hyper_tns,
                              'batched_params_tns': batched_params_tns,
                              'batched_skeletons_tns': batched_skeletons_tns}

exp_dataset = processed_experiments_dict['batched_params_tns'], processed_experiments_dict['batched_skeletons_tns'][0]


########################################################
# TRAINING HYPERPARAMETERS
########################################################
epochs = 150
lr = 'exponential_decay'

num_batches = len(batched_hyper_tns)
tot_shots = batch_size * num_batches
num_test_batches = int(min(max(1, ((0.1 * tot_shots) / batch_size)), 50))
num_train_batches = num_batches - num_test_batches

if lr == 'exponential_decay':
    schedule = optax.exponential_decay(init_value=1e-2, transition_steps = min(500, num_train_batches), decay_rate=0.9, transition_begin = num_train_batches)
    optimizer = optax.adam(learning_rate=schedule)
else:
    optimizer = optax.adam(learning_rate=lr)

# Extract initial params and tensor network structure of the trainable Kraus Operators ("half" of the LPDO, the other half is the transposed conjugate)
kraus_tn = init_lpdo.select('V')

########################################################
# START TRAINING LOOP
########################################################
res = optimize_tn(exp_dataset, 
                  kraus_tn, 
                  optimizer, 
                  true_lpdo=None,
                  epochs=epochs)

########################################################
# SAVE RESULTS
########################################################
def save_results(res: dict, file_name: str, file_folder: str =  'data/optimisation_data/'):
    """
    Save results of the function 'optimize_tn' in json format. 
    Complex values are transformed into strings for json serialisation.
    """
    from datetime import datetime

    make_json_compatible = lambda x: np.array(x, dtype=str).tolist()

    save_dict = {}

    save_dict['batch_size'] = batch_size
    save_dict['build_marginal'] = build_marginals
    save_dict['lr'] = lr
    save_dict['seed'] = seed
    save_dict['noise_type'] = None
    save_dict['data_file'] = data_file
    save_dict['virtual_bd'] = virtual_bd
    save_dict['kraus_bd'] = kraus_bd

    for k in ['train_loss', 'nll', 'distance_to_tp', 'lpdo_trace', 'frob_distance', 'test_loss']:
        save_dict[k] = make_json_compatible(res[k])
    
    save_dict['probs'] = make_json_compatible(res['probs'])

    save_dict['params'] = [{k: np.array(v, dtype=str).tolist() for k, v in p.items()} for p in res['params']]

    with open(file_folder + file_name + f'_seed{seed}.json', 'w') as file:    
        json.dump(save_dict, file, indent = 4)


save_results(res, file_name=data_file, data_folder = "")