import numpy as np
import sys
sys.path.append('..')
from gfast.gfast import GFAST
from pathlib import Path
from synt_exp.synt_src.synthetic_signal import generate_banned_signal_w, SyntheticSubsampledSignal
from itertools import product
from synt_src.synthetic_helper import SyntheticHelper
from gfast.utils import load_data, get_banned_indices_from_qs, get_qs_from_delta, get_qs_from_delta_random, calculate_samples, get_qs_from_delta_sitewise
import time
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
np.random.seed(42)
iter = 0

def random_permutation_matrix(n, iter=iter):
    rng = np.random.RandomState(iter)  
    perm = rng.permutation(n)      
    matrix = np.eye(n)
    perm_matrix = matrix[perm]
    return perm_matrix

def permutation_matrices(qs, num_permutations):
    qs = qs.T
    qs = qs.reshape(-1, 1)
    n = len(qs)
    perm_matrices = []

    for perm in range(num_permutations):
        perm_matrices.append(random_permutation_matrix(n, iter=perm))

    return perm_matrices
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, nargs='+')
    parser.add_argument('--n', type=int)
    parser.add_argument('--q', type=int)
    parser.add_argument('--a', type=int)
    parser.add_argument('--delta', nargs='+', type=int)
    parser.add_argument('--sparsity', type=int)
    parser.add_argument('--iters', type=int)
    parser.add_argument('--snr', type=int)
    parser.add_argument('--removal', nargs='+', type=str)
    parser.add_argument('--num_permutations', type=int)



    """
    Initialize parameters 
    """
    args = parser.parse_args()
    q = args.q
    n = args.n
    b = args.b[0]
    noise_sd = np.sqrt((args.sparsity * args.a**2) / (10**(args.snr / 10)))
    a_min = args.a
    a_max = args.a
    t = n
    n_samples = 10000
    sparsity = args.sparsity
    methods = ['gfast']
    num_repeat = 1
    legend_elements = []
    delays_method_source = "identity"
    delays_method_channel = "identity"
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    unbanned_samples = [calculate_samples(np.array([q] * n), n//b, b, 1) for b in args.b]
    test_args = {
            'n_samples': n_samples
        }
    query_args = {
        "query_method": "simple",
        "delays_method_source": delays_method_source,
        "subsampling_method": "gfast",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "t": t,
        }
    gfast_args = {
            "num_repeat": num_repeat,
            "reconstruct_method_source": delays_method_source,
            "reconstruct_method_channel": delays_method_channel,
            "noise_sd": noise_sd
        }
    removal_policies = args.removal
    threshold = 0.05 # If NMSE is below this threshold, then we consider the experiment to be successful and stop increasing b
    print('Noise SD:', noise_sd)



    """
    Begin synthetic experiments - for each iteration, create a random signal that lives in all removal policies and run all experiments on this specific signal
    """
    data = []
    for i in range(args.iters):
        print('---------------')
        print(f'Iteration {i}')
        print('---------------')
        all_qs_removal = []
        if 'random' in removal_policies:
            all_qs = [get_qs_from_delta_random(delta, q, n) for delta in args.delta]
            all_qs_removal = all_qs_removal + all_qs
        if 'nonuniform' in removal_policies:
            all_qs = [get_qs_from_delta_sitewise(delta, q, n) for delta in args.delta]
            all_qs_removal = all_qs_removal + all_qs
        if 'uniform' in removal_policies:
            all_qs = [get_qs_from_delta(delta, q, n) for delta in args.delta]
            all_qs_removal = all_qs_removal + all_qs
        if all_qs_removal == []:
            raise ValueError(f'No removal policies specified')
        fourier_bounds = np.minimum.reduce(all_qs_removal)
        print(fourier_bounds)
        banned_indices_bounds = get_banned_indices_from_qs(fourier_bounds, q)
        signal_w, locq, strengths = generate_banned_signal_w(n, q, sparsity, a_min, a_max, noise_sd, full=False, banned_indices=banned_indices_bounds)

        max_samples = calculate_samples(np.array([q] * n), n//args.b[-1], args.b[-1], 1)
        full_qs = np.array([q] * n)
        


        for removal in removal_policies:
            print('Removal policy: ', removal)
            
            # Set removal policy for experiments
            base_dir = Path(f'../synt_results/q{q}_n{n}_{removal}_S{sparsity}_snr{args.snr}/')
            if removal == 'random':
                all_qs = [get_qs_from_delta_random(delta, q, n) for delta in args.delta]
            elif removal == 'nonuniform':
                all_qs = [get_qs_from_delta_sitewise(delta, q, n) for delta in args.delta]
            elif removal == 'uniform':
                all_qs = [get_qs_from_delta(delta, q, n) for delta in args.delta]
            else:
                raise ValueError(f'Unknown removal method {removal}')



            """
            Run normal q-SFT - this is equivalent to GFast with no banned alphabets
            """
            print('Normal q-SFT:')
            delta = 0
            samples = []
            nmse = []
            bs = []
            computation_time = []
            banned_indices = get_banned_indices_from_qs(full_qs, q) # Full array of q = normal q-SFT
            signal_params = {
                    "n": n,
                    "q": q,
                    'locq': locq,
                    'strengths': strengths,
                    'banned_indices_toggle': True,
                    'banned_indices': banned_indices,
                    'noise_sd': noise_sd
                }
            N = np.prod(full_qs)
            
            for b1 in range(1, b+1): 
                query_args.update({
                    "num_subsample": n//b1,
                    "b": b1,
                    })
                gfast_args.update({
                    "num_subsample": n//b1,
                    "b": b1,
                })
                newfolder = f'iter{i}_delta{delta}_b{b1}'
                exp_dir = base_dir / newfolder
                exp_dir.mkdir(parents=True, exist_ok=True)

                # Run q-SFT and compute NMSE
                helper = SyntheticHelper(signal_args=signal_params, methods=methods, subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir, subsampling=True)
                start_time = time.time()
                result = helper.compute_model('gfast', gfast_args, report=True, verbosity=0)
                end_time = time.time()
                gwht = result.get("gwht")
                signal_w_diff = signal_w.copy()
                for key in gwht.keys():
                    signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
                nmse_val = np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(signal_w.values())) ** 2)
                nmse.append(nmse_val)        
                samples.append(result['n_samples'])
                bs.append(b1)
                computation_time.append(end_time - start_time)
                print(f'- b = {b1} (samples = {result["n_samples"]}): NMSE = {nmse_val}')
                if nmse_val < threshold:
                    break

            data.append({
                'Removal': removal,
                'Iteration': i,
                'Delta': delta,
                'Samples': samples,
                'NMSE': nmse,
                'b': bs,
                'N': str(N),
                'Computational Time (s)': computation_time,
            })



            """
            Run GFast over different deltas until number of samples matches normal q-SFT
            """
            num_permutations = args.num_permutations
            for qs, delta in zip(all_qs, args.delta):
                print(f'GFast delta {delta}: {qs}')
                samples = []
                nmse = []
                bs = []
                computation_time = []
                differences = [abs(calculate_samples(qs, n//b1, b1, 1) - max_samples) for b1 in range(b, n)]
                nearest_b = np.argmin(differences) + b
                N = np.prod(qs)

                for b1 in range(1, nearest_b + 1):
                    query_args.update({
                        "num_subsample": n//b1,
                        "b": b1,
                        })
                    gfast_args.update({
                        "num_subsample": n//b1,
                        "b": b1,
                    })

                    # Modify the input to get different number of samples
                    nmse_perms = []

                    # Create permutation matrices
                    perm_matrices = permutation_matrices(qs, num_permutations)
                    for j, permutation_matrix in enumerate(perm_matrices):
                        newfolder = f'iter{i}_delta{delta}_b{b1}_perm{j}'
                        exp_dir = base_dir / newfolder
                        exp_dir.mkdir(parents=True, exist_ok=True)

                        perm_qs = (permutation_matrix @ qs.T).astype(int).T
                        print(perm_qs)
                        perm_locq = (permutation_matrix @ locq)
                        banned_indices = get_banned_indices_from_qs(perm_qs, q)
                        signal_params.update({
                            'banned_indices': banned_indices,
                            'locq': perm_locq,
                        })



                        # Run GFast and compute NMSE
                        helper = SyntheticHelper(signal_args=signal_params, methods=methods, subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir, subsampling=True)
                        start_time = time.time()
                        result = helper.compute_model('gfast', gfast_args, report=True, verbosity=0)
                        end_time = time.time()
                        samples.append(result['n_samples'])
                        bs.append(b1)
                        computation_time.append(end_time - start_time)
                        signal_w_perm = dict(zip(list(map(tuple, perm_locq.T)), strengths))
                        gwht = result.get("gwht")
                        signal_w_diff = signal_w_perm.copy()
                        for key in gwht.keys():
                            signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
                        nmse_val = np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(signal_w_perm.values())) ** 2)
                        nmse_perms.append(nmse_val)
                        print(f'- b = {b1}, perm = {j} (samples = {result["n_samples"]}): NMSE = {nmse_val}')

                    nmse = nmse + nmse_perms
                    if np.mean(nmse_perms) < threshold:
                        break
                    

                data.append({
                    'Removal': removal,
                    'Iteration': i,
                    'Delta': delta,
                    'Samples': samples,
                    'NMSE': nmse,
                    'b': bs,
                    'N': str(N),
                    'Computational Time (s)': computation_time,
                })
                
    df = pd.DataFrame(data)
    df.to_csv(f'../synt_results/q{q}_n{n}_results_S{sparsity}_snr{args.snr}.csv')
            