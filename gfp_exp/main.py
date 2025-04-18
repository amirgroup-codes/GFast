"""
Main script to run the GFast model on the GFP model.
"""
import random
import pandas as pd
import pickle
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import subprocess
from src.helper import Helper
from gfp_utils import read_fasta, select_aas, load_model, get_random_test_samples, qary_to_aa_encoding
from gfast.utils import get_qs, save_data4, find_matrix_indices, test_nmse, calculate_samples
from compute_samples import compute_scores


def parse_args():
    parser = argparse.ArgumentParser(description="Get q-ary indices.")
    parser.add_argument("--n", nargs='+', type=int, required=True)
    parser.add_argument("--b", nargs='+', type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_weights = np.load('model/model_weights.npy')
    sequence = read_fasta('model/avGFP.fasta')
    sequence = sequence[:-1]
    q = 20
    ns = args.n
    bs = args.b
    noise_sd = args.threshold
    t = 3
    num_repeat = 20
    delays_method_source = "identity"
    delays_method_channel = "nr"
    hyperparam = True
    model = load_model('model/mlp.pt')
    # Select the AAs that are above the threshold
    threshold = args.threshold
    banned_indices = select_aas(sequence, threshold, model_weights)



    """
    Run GFast
    """
    data = []
    for (b, n) in zip(bs, ns):
        base_dir = Path(f'../gfp_results/q{q}_n{n}_{delays_method_channel}_{threshold}/')
        os.makedirs(base_dir, exist_ok=True)
        unbanned_samples = [calculate_samples(np.array([q] * n), n//b, b, 1) for b in args.b] 
        full_qs = np.array([q] * n)

        # Remove banned indices if they exist in the GFP dataset - this is so we can test against the ground truth data
        banned_indices_n = dict(itertools.islice(banned_indices.items(), n))
        with open(f'{base_dir}/banned_indices_n.pkl', 'wb') as f:
            pickle.dump(banned_indices_n, f)
        qs = get_qs(q, n, banned_indices_n)

        # Keep 
        min_length = min(len(indices) for indices in banned_indices_n.values())
        # Randomly trim lists to the minimum length
        trimmed_banned_indices_n = {
            key: random.sample(indices, min_length)
            for key, indices in banned_indices_n.items()
        }
        with open(f'{base_dir}/qsft_banned_indices_n.pkl', 'wb') as f:
            pickle.dump(trimmed_banned_indices_n, f)

        # We want to use the same test indices across all different runs, so we will override 
        # the generated test helper indices by specifying the same indices for all runs
        test_samples = get_random_test_samples(q, n, banned_indices_n) # We need to convert this to be amino acid specific
        test_samples_nmse = get_random_test_samples(q, n, banned_indices_n) 
        scores_nmse = compute_scores(n, sequence, test_samples_nmse, banned_indices_n, banned_indices_toggle=True)

        qary_to_aa_dict = qary_to_aa_encoding(banned_indices_n)
        qary_to_aa_dict_qsft = qary_to_aa_encoding(trimmed_banned_indices_n)
        reversed_qary_to_aa_dict_qsft = {
            i: {v: k for k, v in qary_to_aa_dict_qsft[i].items()}
            for i in qary_to_aa_dict_qsft
        }
        new_dict = {}
        for position in qary_to_aa_dict:
            new_dict[position] = {
                key: reversed_qary_to_aa_dict_qsft[position].get(value)
                for key, value in qary_to_aa_dict[position].items()
            }
        new_qs = get_qs(q, n, trimmed_banned_indices_n)
    


        """
        Normal q-SFT - this is equivalent to GFast with no banned alphabets
        """
        print('Normal q-SFT:')
        delta = 0
        for b1 in range(1, b+1): 
            for d in range(5, num_repeat+1):
                # Generate indices for sampling
                num_subsample = n//b1
                newfolder = f'delta{delta}_b{b1}_d{d}'
                exp_dir = base_dir / newfolder
                exp_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    "python", "get_qary_indices.py", 
                    "--q", str(q), 
                    "--n", str(n), 
                    "--delta", str(delta), 
                    "--b", str(b1), 
                    "--num_subsample", str(num_subsample), 
                    "--num_repeat", str(d), 
                    "--banned_indices_path", f"{base_dir}/banned_indices_n.pkl", 
                    "--exp_dir", str(exp_dir),
                    "--banned_indices_toggle", 'False',
                    "--delays_method_channel", delays_method_channel
                ])
                subprocess.run([
                    "python", "get_qary_indices.py", 
                    "--q", str(q), 
                    "--n", str(n), 
                    "--delta", str(delta), 
                    "--b", str(b1), 
                    "--num_subsample", str(num_subsample), 
                    "--num_repeat", str(d), 
                    "--banned_indices_path", f"{base_dir}/qsft_banned_indices_n.pkl", 
                    "--exp_dir", str(exp_dir),
                    "--banned_indices_toggle", 'True',
                    "--delays_method_channel", delays_method_channel
                ])
                # Change test samples to have the same indices as the gfast samples
                test_file_indices = Path(f"{exp_dir}/test/signal_t_queryindices.pickle")
                test_file_qaryindices = Path(f"{exp_dir}/test/signal_t_query_qaryindices.pickle")
                converted_test_samples = np.array([
                    [new_dict[i][value] for value in test_samples[i]]
                    for i in range(test_samples.shape[0])
                ])
                convert_test_samples_qary_indices = find_matrix_indices(converted_test_samples, new_qs)
                save_data4(convert_test_samples_qary_indices, test_file_qaryindices)
                save_data4(converted_test_samples, test_file_indices)

                # Sample training points from model
                subprocess.run([
                    "python", "compute_samples.py", 
                    "--n", str(n), 
                    "--num_subsample", str(num_subsample), 
                    "--num_repeat", str(d), 
                    "--exp_dir", str(exp_dir),
                    "--banned_indices_toggle", 'True',
                    "--banned_indices_path", f"{base_dir}/qsft_banned_indices_n.pkl"
                ])

                # Run GFast
                result = subprocess.Popen([
                    "python", "run_gfast.py", 
                    "--q", str(q), 
                    "--n", str(n), 
                    "--b", str(b1),
                    "--num_subsample", str(num_subsample), 
                    "--num_repeat", str(d), 
                    "--exp_dir", str(exp_dir),
                    "--banned_indices_toggle", 'True',
                    "--banned_indices_path", f"{base_dir}/qsft_banned_indices_n.pkl",
                    "--delays_method_source", delays_method_source,
                    "--delays_method_channel", delays_method_channel,
                    "--hyperparam", str(hyperparam)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
                
                captured_output = []
                for line in result.stdout:
                    print(line, end="")
                    captured_output.append(line.strip())
                result.wait()

                # Validate on new test data
                converted_test_samples_nmse = np.array([
                    [new_dict[i][value] for value in test_samples_nmse[i]]
                    for i in range(test_samples_nmse.shape[0])
                ])
                gwht = np.load(f'{exp_dir}/gwht.pkl', allow_pickle=True)
                nmse = test_nmse(converted_test_samples_nmse, scores_nmse, gwht, q, n, exp_dir, trimmed_banned_indices_n)
                print(f"NMSE: {nmse}")
                print('----------')



                data.append({
                    'Method': 'q-SFT',
                    'Samples': calculate_samples(np.array([q] * n), n//b1, b1, d),
                    'b': b1,
                    'd': d,
                    'nmse': nmse
                })

            



        """
        GFast
        """
        print('GFast')
        qs = get_qs(q, n, banned_indices_n)
        print(qs)
        for b1 in range(1, b+1):
            for d in range(5, num_repeat+1):
                # Generate indices for sampling
                num_subsample = n//b1
                newfolder = f'delta{delta}_b{b1}_d{d}'
                exp_dir = base_dir / newfolder
                exp_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    "python", "get_qary_indices.py", 
                    "--q", str(q), 
                    "--n", str(n), 
                    "--delta", str(delta), 
                    "--b", str(b1), 
                    "--num_subsample", str(num_subsample), 
                    "--num_repeat", str(d), 
                    "--banned_indices_path", f"{base_dir}/banned_indices_n.pkl", 
                    "--exp_dir", str(exp_dir),
                    "--banned_indices_toggle", 'True',
                    "--delays_method_channel", delays_method_channel
                ])
                test_file_indices = Path(f"{exp_dir}/test/signal_t_queryindices.pickle")
                test_file_qaryindices = Path(f"{exp_dir}/test/signal_t_query_qaryindices.pickle")
                test_samples_qary_indices = find_matrix_indices(test_samples, qs)
                save_data4(test_samples_qary_indices, test_file_qaryindices)
                save_data4(test_samples, test_file_indices)

                # Sample training points from model
                subprocess.run([
                    "python", "compute_samples.py", 
                    "--n", str(n), 
                    "--num_subsample", str(num_subsample), 
                    "--num_repeat", str(d), 
                    "--exp_dir", str(exp_dir),
                    "--banned_indices_toggle", 'True',
                    "--banned_indices_path", f"{base_dir}/banned_indices_n.pkl"
                ])

                # Run GFast
                result = subprocess.Popen([
                    "python", "run_gfast.py", 
                    "--q", str(q), 
                    "--n", str(n), 
                    "--b", str(b1),
                    "--num_subsample", str(num_subsample), 
                    "--num_repeat", str(d), 
                    "--exp_dir", str(exp_dir),
                    "--banned_indices_toggle", 'True',
                    "--banned_indices_path", f"{base_dir}/banned_indices_n.pkl",
                    "--delays_method_source", delays_method_source,
                    "--delays_method_channel", delays_method_channel,
                    "--hyperparam", str(hyperparam),
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                captured_output = []
                for line in result.stdout:
                    print(line, end="")
                    captured_output.append(line.strip())
                result.wait()

                # Validate on new test data
                gwht = np.load(f'{exp_dir}/gwht.pkl', allow_pickle=True)
                nmse = test_nmse(test_samples_nmse, scores_nmse, gwht, q, n, exp_dir, banned_indices_n)
                print(f"NMSE: {nmse}")
                print('----------')

                data.append({
                    'Method': 'GFast',
                    'Samples': calculate_samples(qs, n//b1, b1, d),
                    'b': b1,
                    'd': d,
                    'nmse': nmse
                })

        df = pd.DataFrame(data)
        df.to_csv(f'../gfp_results/q{q}_n{n}_{delays_method_channel}_{threshold}.csv')

