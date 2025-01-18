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
from math import floor
import itertools
import subprocess
from src.helper import Helper
from gfp_utils import read_fasta, select_aas, convert_encoding, aa_indices, sample_model, one_hot_encode, load_model, get_random_test_samples, qary_to_aa_encoding
from gfast.plot_utils import calculate_samples
from gfast.utils import load_data, save_data, get_qs, summarize_results, process_stdout, qary_vector_banned, dec_to_qary_vec, save_data4, qary_vec_to_dec, find_matrix_indices, decimal_banned


def parse_args():
    parser = argparse.ArgumentParser(description="Get q-ary indices.")
    parser.add_argument("--n", nargs='+', type=int, required=True)
    parser.add_argument("--b", nargs='+', type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    # parser.add_argument("--num_repeat", type=int, required=True)
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
    num_repeat = 20#int(args.num_repeat)
    delays_method_source = "identity"
    delays_method_channel = "nso"
    hyperparam = True
    model = load_model('model/mlp.pt')
    # Select the AAs that are above the threshold
    threshold = args.threshold
    banned_indices = select_aas(sequence, threshold, model_weights)
    #print(banned_indices)



    """
    Run GFast
    """
    data = []
    for (b, n) in zip(bs, ns):
        base_dir = Path(f'../gfp_results/q{q}_n{n}_{delays_method_channel}_{threshold}/')
        os.makedirs(base_dir, exist_ok=True)
        unbanned_samples = [calculate_samples(np.array([q] * n), n//b, b, 1) for b in args.b] 
        full_qs = np.array([q] * n)

        # df_val = pd.read_csv('GFP.csv')
        # comparison_seq = sequence[n+1:]
        # print(len(df_val))
        # df_val = df_val[df_val['seq'] == comparison_seq[:len(df_val['seq'].iloc[0])]]
        # print(len(df_val))
        # df_val['seq'] = df_val['seq'].str[0:n] 
        # y = df_val['medianBrightness'].to_numpy()
        # qsft_set = np.array([[aa_dict[aa] for aa in seq] for seq in df_val['seq']])
        # # Remove banned indices if they exist in the GFP dataset - this is so we can test against the ground truth data
        banned_indices_n = dict(itertools.islice(banned_indices.items(), n))
        # min_length = 16
        # print(banned_indices_n)
        # banned_indices_n = ensure_length_16(banned_indices_n, q)
        # for pos in range(qsft_set.shape[1]):
        #     amino_acids_at_pos = set(qsft_set[:, pos])
        #     banned_indices_n[pos] = [num for num in banned_indices_n[pos] if num not in amino_acids_at_pos]
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
        # print(trimmed_banned_indices_n)
        # print(banned_indices_n)
        # print(banned_indices_n)
        # print(qs)
        # print(trimmed_banned_indices_n)
        with open(f'{base_dir}/qsft_banned_indices_n.pkl', 'wb') as f:
            pickle.dump(trimmed_banned_indices_n, f)
        # qs_trimmed = get_qs(q, n, trimmed_banned_indices_n)
        # print(qs_trimmed)

        # We want to use the same test indices across all different runs, so we will override 
        # the generated test helper indices by specifying the same indices for all runs
        test_samples = get_random_test_samples(q, n, banned_indices_n) # We need to convert this to be amino acid specific
        qary_to_aa_dict = qary_to_aa_encoding(banned_indices_n)
        qary_to_aa_dict_qsft = qary_to_aa_encoding(trimmed_banned_indices_n)
        reversed_qary_to_aa_dict_qsft = {
            i: {v: k for k, v in qary_to_aa_dict_qsft[i].items()}
            for i in qary_to_aa_dict_qsft
        }
        # qary_to_aa_dict_qsft_reversed = {v: k for k, v in qary_to_aa_dict_qsft.items()}
        new_dict = {}
        for position in qary_to_aa_dict:
            new_dict[position] = {
                key: reversed_qary_to_aa_dict_qsft[position].get(value)
                for key, value in qary_to_aa_dict[position].items()
            }
        # print(qary_to_aa_dict)
        # print(qary_to_aa_dict_qsft)
        # print(new_dict)
        new_qs = get_qs(q, n, trimmed_banned_indices_n)
        # Create GFast dataset now. This is the same dataset as the GFP dataset + q-SFT dataset, but we need to convert the q-ary encoding 
        # Prepare test set of validation proteins after hyperparameter tuning. This involves converting the experimental protein sequences into their respective q-ary encodings
        # aa_to_qary_dict = {pos: {v: k for k, v in mapping.items()} for pos, mapping in qary_to_aa_dict.items()}
        # gfast_set = np.array([
        #     [aa_to_qary_dict[pos][aa_num] for pos, aa_num in enumerate(row)]
        #     for row in qsft_set
        # ])
        # print(gfast_set.shape)
        # test = aa_indices(n, sequence, gfast_set, banned_indices=banned_indices_n, banned_indices_toggle='True')
        # print(''.join(test[10])[0:n], y[10])
        # print(df_val['seq'][10], df_val['medianBrightness'][10])
        # print(qsft_set[0:1, 0:n])
        # print(qary_to_aa_dict)
        # print(gfast_set[0:1, 0:n])

        # aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
        # num_encoding = {v: k for k, v in aa_dict.items()}
        # # Adjust amino acid dictionary to only include the amino acids that are not in the banned indices
        # site_aa_encoding = {}
        # for key, value in banned_indices.items():
        #     encoding_reduced = {k: v for k, v in num_encoding.items() if k not in value}
        #     site_aa_encoding[key] = {new_key: v for new_key, (_, v) in enumerate(encoding_reduced.items())}
        # print(site_aa_encoding)
        # #KUNAL: would be easier to do this editing input_signal_subsampled.py but i dont have permissions
        # #generate the test indices ONCE for everything per n
        # n_samples = 5
        # max_n = np.prod(get_qs(q, n, banned_indices_n))
        # base_inds_dec = [floor(random.uniform(0, 1) * max_n) for _ in range(n_samples)]
        # base_inds_dec = np.unique(np.array(base_inds_dec, dtype=int))
        # print("base inds dec", base_inds_dec)
        # #now that we have this, we can just use it for all of the test indices (after converting encoding)
        # query_indices_mixedradix = np.array(qary_vector_banned(base_inds_dec, get_qs(q, n, banned_indices_n))).T
        # print("QUERY INDICES MIXED RADIX", query_indices_mixedradix)
        # # print("AA mixed shape", np.shape(query_indices_mixedradix))
        # # aas_from_mixed = aa_indices(n, sequence, query_indices_mixedradix, banned_indices_toggle=True, banned_indices=banned_indices_n)
        # # print("AAs from mixed", aas_from_mixed)
        # # one_hot_encoded_sequences = one_hot_encode(aas_from_mixed)
        # # samples = sample_model(model, one_hot_encoded_sequences, batch_size=512)
        # # print("AA samples banned" , samples)
        # query_indices_regular = convert_encoding(query_indices_mixedradix, banned_indices_n)
        # print("QUERY INDICES REGULAR", query_indices_regular)
        # # print("AA regular shape", np.shape(query_indices_regular))
        # decimal_query_indices = qary_vec_to_dec(query_indices_regular, q)
        # # aas_from_regular = aa_indices(n, sequence, query_indices_regular, banned_indices_toggle=False)
        # # one_hot_encoded_sequences = one_hot_encode(aas_from_regular)
        # # samples = sample_model(model, one_hot_encoded_sequences, batch_size=512)
        # # print("AA samples regular" , samples)

        """
        Normal q-SFT - this is equivalent to GFast with no banned alphabets
        """
        print('Normal q-SFT:')
        delta = 0
        for b1 in range(1, b+1): 
            for d in range(5, num_repeat+1):
            # for d in range(4, num_repeat + 1, 4):
                # Generate indices for sampling
                num_subsample = n//b1
                newfolder = f'delta{delta}_b{b1}_d{d}'
                exp_dir = base_dir / newfolder
                exp_dir.mkdir(parents=True, exist_ok=True)
                # subprocess.run([
                #     "python", "get_qary_indices.py", 
                #     "--q", str(q), 
                #     "--n", str(n), 
                #     "--delta", str(delta), 
                #     "--b", str(b1), 
                #     "--num_subsample", str(num_subsample), 
                #     "--num_repeat", str(d), 
                #     "--banned_indices_path", f"{base_dir}/banned_indices_n.pkl", 
                #     "--exp_dir", str(exp_dir),
                #     "--banned_indices_toggle", 'False',
                #     "--delays_method_channel", delays_method_channel
                # ])
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
                # converted_test_samples = np.array([
                #     [qary_to_aa_dict[i][value] for value in test_samples[i]]
                #     for i in range(test_samples.shape[0])
                # ])
                converted_test_samples = np.array([
                    [new_dict[i][value] for value in test_samples[i]]
                    for i in range(test_samples.shape[0])
                ])
                # print(test_samples)
                # print(new_dict)
                # print(converted_test_samples)
                # convert_test_samples_qary_indices = qary_vec_to_dec(converted_test_samples, q)
                convert_test_samples_qary_indices = find_matrix_indices(converted_test_samples, new_qs)
                # print(convert_test_samples_qary_indices)
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
                # subprocess.run([
                #     "python", "compute_samples.py", 
                #     "--n", str(n), 
                #     "--num_subsample", str(num_subsample), 
                #     "--num_repeat", str(d), 
                #     "--exp_dir", str(exp_dir),
                #     "--banned_indices_toggle", 'False',
                #     "--banned_indices_path", f"{base_dir}/banned_indices_n.pkl"
                # ])

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
                # result = subprocess.Popen([
                #     "python", "run_gfast.py", 
                #     "--q", str(q), 
                #     "--n", str(n), 
                #     "--b", str(b1),
                #     "--num_subsample", str(num_subsample), 
                #     "--num_repeat", str(d), 
                #     "--exp_dir", str(exp_dir),
                #     "--banned_indices_toggle", 'False',
                #     "--banned_indices_path", f"{base_dir}/banned_indices_n.pkl",
                #     "--delays_method_source", delays_method_source,
                #     "--delays_method_channel", delays_method_channel,
                #     "--hyperparam", str(hyperparam)
                # ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
                
                captured_output = []
                for line in result.stdout:
                    print(line, end="")
                    captured_output.append(line.strip())
                result.wait()
                nmse = process_stdout(captured_output)

                # # Validate on the ground truth data
                # gwht = np.load(f'{exp_dir}/gwht.pkl', allow_pickle=True)
                # protein_nmse = test_nmse(qsft_set, y, gwht, q, n, {})
                # print(f"Protein NMSE: {protein_nmse}")
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
        delta = 1
        for b1 in range(1, b+1): 
            for d in range(5, num_repeat+1):
            # for d in range(4, num_repeat + 1, 4):
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
                    "--hyperparam", str(hyperparam)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                captured_output = []
                for line in result.stdout:
                    print(line, end="")
                    captured_output.append(line.strip())
                result.wait()
                nmse = process_stdout(captured_output)

                # # Validate on the ground truth data
                # gwht = np.load(f'{exp_dir}/gwht.pkl', allow_pickle=True)
                # protein_nmse = test_nmse(gfast_set, y, gwht, q, n, banned_indices_n)
                # print(f"Protein NMSE: {protein_nmse}")
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

            # for qs, delta in zip(all_qs, args.delta):
            #     print(f'GFast delta {delta}:')
            #     samples = []
            #     nmse = []
            #     differences = [abs(calculate_samples(qs, n//b1, b1, 1) - max_samples) for b1 in range(b, n)]
            #     nearest_b = np.argmin(differences) + b

            #     for b1 in range(1, nearest_b + 1):
            #         query_args.update({
            #             "num_subsample": n//b1,
            #             "b": b1,
            #             })
            #         gfast_args.update({
            #             "num_subsample": n//b1,
            #             "b": b1,
            #         })

            #         # Modify the input to get different number of samples
            #         nmse_perms = []

            #         # Create permutation matrices
            #         # print(qs.shape)
            #         perm_matrices = permutation_matrices(qs, num_permutations)
            #         for j, permutation_matrix in enumerate(perm_matrices):
            #             newfolder = f'iter{i}_delta{delta}_b{b1}_perm{j}'
            #             exp_dir = base_dir / newfolder
            #             exp_dir.mkdir(parents=True, exist_ok=True)

            #             perm_qs = (permutation_matrix @ qs.T).astype(int).T
            #             perm_locq = (permutation_matrix @ locq)
            #             banned_indices = get_banned_indices_from_qs(perm_qs, q)
            #             signal_params.update({
            #                 'banned_indices': banned_indices,
            #                 'locq': perm_locq,
            #             })



            #             # Run GFast and compute NMSE
            #             helper = SyntheticHelper(signal_args=signal_params, methods=methods, subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir, subsampling=True)
            #             result = helper.compute_model('gfast', gfast_args, report=True, verbosity=0)
            #             samples.append(result['n_samples'])
            #             nmse_val = helper.test_model('gfast', beta=result['gwht'])
            #             if isinstance(nmse_val, tuple):
            #                 nmse_val = nmse_val[0]
            #                 nmse_perms.append(nmse_val)
            #             else:
            #                 nmse_perms.append(nmse_val)
            #             print(f'- b = {b1}, perm = {j} (samples = {result["n_samples"]}): NMSE = {nmse_val}')
            #             # print(permutation_matrix)
            #             # print(perm_qs, qs)

            #         nmse = nmse + nmse_perms
            #         if np.mean(nmse_perms) < threshold:
            #             break


            # query_args = {
            #     "query_method": "simple",
            #     "num_subsample": num_subsample,
            #     "delays_method_source": delays_method_source,
            #     "subsampling_method": "gfast",
            #     "delays_method_channel": delays_method_channel,
            #     "num_repeat": num_repeat,
            #     "b": b1,
            #     "t": t,
            #     "folder": exp_dir 
            # }
            # signal_args = {
            #                 "n":n,
            #                 "q":q,
            #                 "noise_sd":noise_sd,
            #                 "query_args":query_args,
            #                 "t": t,
            #                 'banned_indices_toggle': False,
            #                 'banned_indices': banned_indices_qsft
            #                 }
            # test_args = {
            #         "n_samples": 10000
            #     }
            
            # helper = Helper(signal_args=signal_args, methods=["gfast"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
            # model_kwargs = {}
            # model_kwargs["num_subsample"] = num_subsample
            # model_kwargs["num_repeat"] = num_repeat
            # model_kwargs["b"] = b1
            # test_kwargs = {}
            # model_kwargs["n_samples"] = num_subsample * (helper.q ** b1) * num_repeat * (helper.n + 1)
            # model_kwargs["noise_sd"] = noise_sd
            # model_result = helper.compute_model(method="gfast", model_kwargs=model_kwargs, report=True, verbosity=0)
            # test_kwargs["beta"] = model_result.get("gwht")
            # nmse, r2_value = helper.test_model("gfast", **test_kwargs)
            # gwht = model_result.get("gwht")
            # locations = model_result.get("locations")
            # n_used = model_result.get("n_samples")
            # avg_hamming_weight = model_result.get("avg_hamming_weight")
            # max_hamming_weight = model_result.get("max_hamming_weight")
            # print(f"R^2: {r2_value}, NMSE: {nmse}")

            # qs = get_qs(q, n, banned_indices_qsft)
            # summarize_results(locations, gwht, q, n, b1, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, exp_dir, args, qs)





            # query_args = {
            #     "query_method": "generate_samples",
            #     "method": "generate_samples",
            #     "num_subsample": num_subsample,
            #     "num_repeat": num_repeat,
            #     "b": b1,
            #     "folder": exp_dir 
            # }
            # signal_args = {
            #     "n":n,
            #     "q":q,
            #     "query_args":query_args,
            #     "len_seq":n,
            #     "banned_indices_toggle": True,
            #     "banned_indices": banned_indices_n
            #     }
            # test_args = {
            #         "n_samples": 10000,
            #         "method": "generate_samples"
            #     }
            # file_path = os.path.join(exp_dir, 'train', 'samples')
            # if not os.path.exists(file_path):
            #     os.makedirs(file_path, exist_ok=True)
            # file_path = os.path.join(exp_dir, 'test')
            # if not os.path.exists(file_path):
            #     os.makedirs(file_path, exist_ok=True)

            # try:
            #     helper = Helper(signal_args=signal_args, methods=["gfast"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
            # except ScriptCompleted as e:
            #     print(f"Handled completion of input_signal_subsampled.py: {e}")

