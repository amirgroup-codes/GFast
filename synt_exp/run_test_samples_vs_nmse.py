import numpy as np
import sys
sys.path.append('..')
from gfast.gfast import GFAST
from pathlib import Path
from synt_exp.synt_src.synthetic_signal import generate_banned_signal_w, SyntheticSubsampledSignal
from itertools import product
from synt_src.synthetic_helper import SyntheticHelper
from gfast.plot_utils import get_banned_indices_from_qs, get_qs_from_delta, get_qs_from_delta_random, calculate_samples, get_qs_from_delta_sitewise
from gfast.utils import load_data
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

    # # Create the largest one
    # large_qs = np.sort(qs).reshape(-1, 1)
    # P = np.zeros((n, n), dtype=int)
    # for i, val in enumerate(large_qs):
    #     j = np.where(qs == val)[0][0]
    #     P[i, j] = 1
    # # print(qs.T.shape, large_qs.shape, np.linalg.pinv(qs).shape)
    # # P = large_qs @ np.linalg.pinv(qs)
    # perm_matrices.append(np.array(P).astype(int))

    # # Create smallest one 
    # small_qs = []
    # start, end = 0, len(large_qs) - 1
    # while start <= end:
    #     small_qs.append(large_qs[start])
    #     if start != end:  # Avoid adding the middle element twice in odd-length arrays
    #         small_qs.append(large_qs[end])
    #     start += 1
    #     end -= 1
    # P = np.array(small_qs) @ np.linalg.pinv(qs)
    # P = np.zeros((n, n), dtype=int)
    # for i, val in enumerate(small_qs):
    #     j = np.where(qs == val)[0][0]
    #     P[i, j] = 1
    # perm_matrices.append(np.array(P).astype(int))

    # Rest can be random
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
                # nmse_val = helper.test_model('gfast', beta=result['gwht'])
                # if isinstance(nmse_val, tuple):
                #     nmse_val = nmse_val[0]
                #     nmse.append(nmse_val)
                # else:
                #     nmse.append(nmse_val)
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
                    # print(qs.shape)
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
                        # nmse_val2 = helper.test_model('gfast', beta=result['gwht'])
                        # print(nmse_val, nmse_val2[0])
                        # print(gwht)
                        # print(signal_w_perm)
                        # if isinstance(nmse_val, tuple):
                        #     nmse_val = nmse_val[0]
                        #     nmse_perms.append(nmse_val)
                        # else:
                        #     nmse_perms.append(nmse_val)
                        print(f'- b = {b1}, perm = {j} (samples = {result["n_samples"]}): NMSE = {nmse_val}')
                        # print(permutation_matrix)
                        # print(perm_qs, qs)

                    nmse = nmse + nmse_perms
                    if np.mean(nmse_perms) < threshold:
                        break
                    
                    # for perm in range(num_permutations):
                    #     newfolder = f'iter{i}_delta{delta}_b{b1}_perm{perm}'
                    #     exp_dir = base_dir / newfolder
                    #     exp_dir.mkdir(parents=True, exist_ok=True)

                    #     permutation_matrix = random_permutation_matrix(n, iter=iter)
                    #     iter += 1
                    #     perm_qs = (permutation_matrix @ qs.T).astype(int).T
                    #     perm_locq = (permutation_matrix @ locq)
                    #     banned_indices = get_banned_indices_from_qs(perm_qs, q)
                    #     signal_params.update({
                    #         'banned_indices': banned_indices,
                    #         'locq': perm_locq,
                    #     })



                    #     # Run GFast and compute NMSE
                    #     helper = SyntheticHelper(signal_args=signal_params, methods=methods, subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir, subsampling=True)
                    #     result = helper.compute_model('gfast', gfast_args, report=True, verbosity=0)
                    #     samples.append(result['n_samples'])
                    #     nmse_val = helper.test_model('gfast', beta=result['gwht'])
                    #     if isinstance(nmse_val, tuple):
                    #         nmse_val = nmse_val[0]
                    #         nmse_perms.append(nmse_val)
                    #     else:
                    #         nmse_perms.append(nmse_val)
                    #     print(f'- b = {b1}, perm = {perm} (samples = {result["n_samples"]}): NMSE = {nmse_val}')

                    # nmse = nmse + nmse_perms
                    # if min(nmse_perms) < threshold:
                    #     break

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
            






#         """
#         For each set of banned indices, 
#         """
#         for color, qs, delta in zip(colors, all_qs, args.delta):
#             banned_indices = get_banned_indices_from_qs(qs, q)
#             samples = []
#             nmse = []
#             r2 = []
#             differences = [abs(calculate_samples(qs, n//b, b, 1) - max_samples) for b in range(args.b[-1], n)]
#             nearest_b = np.argmin(differences) + args.b[-1]
#             signal_params = {
#                 "n": n,
#                 "q": q,
#                 'locq': locq,
#                 'strengths': strengths,
#                 'banned_indices_toggle': True,
#                 'banned_indices': banned_indices,
#                 'noise_sd': noise_sd
#             }
#             print("LAST B: ", nearest_b+1)
#             for b1 in range(args.b[0], nearest_b):
#                 print(f"iter {i} delta {delta} bval {b1}")
#                 query_args = {
#                     "query_method": "simple",
#                     "num_subsample": n//b1,
#                     "delays_method_source": delays_method_source,
#                     "subsampling_method": "gfast",
#                     "delays_method_channel": delays_method_channel,
#                     "num_repeat": num_repeat,
#                     "b": b1,
#                     "t": t,
#                     }
#                 test_args = {
#                         'n_samples': n_samples
#                     }
#                 gfast_args = {
#                     "num_subsample": n//b1,
#                     "num_repeat": num_repeat,
#                     "reconstruct_method_source": delays_method_source,
#                     "reconstruct_method_channel": delays_method_channel,
#                     "b": b1,
#                     "noise_sd": noise_sd
#                 }
#                 newfolder = f'it{i}d{delta}b{b1}'
#                 exp_dir = base_dir / newfolder
#                 exp_dir.mkdir(parents=True, exist_ok=True)
#                 helper = SyntheticHelper(signal_args=signal_params, methods=methods, subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir, subsampling=True)
#                 result = helper.compute_model('gfast', gfast_args, report=True, verbosity=0)
#                 samples.append(result['n_samples'])
#                 nmse_val = helper.test_model('gfast', beta=result['gwht'])
#                 print(nmse_val)
#                 if isinstance(nmse_val, tuple):
#                     r2_val = nmse_val[1]
#                     nmse_val = nmse_val[0]
#                     nmse.append(nmse_val)
#                     r2.append(r2_val)
#                 data.append({
#                     'Delta': delta,
#                     'Samples': samples,
#                     'NMSE': nmse,
#                     'R2': r2
#                 })

#                 # gwht = result['gwht']
#                 # signal_w_diff = signal_w.copy()
#                 # for key in gwht.keys():
#                 #     signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
#                 # # print('gwht diff', signal_w_diff)
#                 # nmse_val2 = np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(signal_w.values())) ** 2)
#                 # print('gwht test nmse', nmse_val2)
#                 # # print('gwht ground ', test_signal.signal_w)

#             banned_pairs = sorted(zip(samples, nmse))
#             samples_banned, nmse_b = zip(*banned_pairs)
#             samples_banned = list(samples_banned)
#             nmse_b = list(nmse_b)   
#             label = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{delta} Alphabet\n Removed')    
#             legend_elements.append(label)
#             ax.scatter(samples, nmse, color=color)
#             plt.plot(samples, nmse, color=color)
# df = pd.DataFrame(data)

# ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
# ax.set_xscale('log')
# ax.set_xlabel('Samples used')
# ax.set_ylabel('NMSE')
# ax.set_title(rf'q = {q}, n = {n}, $S$ = {args.sparsity}, SNR = {args.snr}' , fontsize=16, pad=20)
# ax.grid(True)
# #plt.legend()
# plt.tight_layout()
# plt.subplots_adjust(right=0.85, top=0.9)
# #df.to_pickle(base_dir / 'data.pkl')
# df.to_csv(f'../synt_results/q{q}_n{n}_{args.removal}.csv')
# fig.savefig(f'../synt_results/q{q}_n{n}_{args.removal}.png')
# plt.show()
