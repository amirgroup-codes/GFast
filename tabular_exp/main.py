"""
Main script to run the GFast model on tabular data.
"""
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from gfast.commands import get_qary_indices, compute_gfast_samples, run_gfast
from utils import ModelScorer, load_model
import joblib
import torch
import argparse
from pathlib import Path
from gfp_exp.gfp_utils import get_random_test_samples
from gfast.utils import get_banned_indices_from_qs, test_nmse, load_data, calculate_samples
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Get q-ary indices.")
    parser.add_argument("--b", type=int, required=True)
    return parser.parse_args()

args = parse_args()
df_datasets = pd.read_csv("data/datasets.csv")
datasets = df_datasets['Dataset'].values
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

def calculate_samples_qary(exp_dir, n, b, d, scorer):

    total_samples = 0
    C = n // b
    for c in range(C):
        for d in range(d):
            indices_sampled = os.path.join(exp_dir, "train/samples/M{}_D{}_queryindices.pickle".format(b, d))
            indices_sampled = np.concatenate(load_data(indices_sampled))
            total_samples += scorer.get_samples_qary(indices_sampled)
    return total_samples
    


bs = np.arange(1, int(args.b) + 1)

# Run GFast on heart disease dataset
for dataset in datasets:
    print(f"{dataset}")
    print('Running GFast')
    qs = np.load(f'model/{dataset}_qs.npy')
    ds = np.arange(1, len(qs) + 1)
    print("qs", qs)
    q = np.max(qs)
    n = len(qs)

    # Compute samples
    model = load_model(f"model/{dataset}_model.pth", np.sum(qs), device)
    scorer = ModelScorer(model, qs, device=device)

    banned_indices = get_banned_indices_from_qs(qs, q)
    test_samples = get_random_test_samples(q, n, banned_indices) # Get test samples for all runs
    test_samples_y = scorer.compute_scores(test_samples.T)

    data = []
    for b in bs:
        for d in ds:

            # Generate indices for sampling
            exp_dir = f"../tabular_results/{dataset}_gfast/q{q}_n{n}_b{b}_d{d}/"
            if not os.path.exists(f'{exp_dir}nmse.npy'):
                params = {
                    "qs": qs,
                    "n": n,
                    "b": b,
                    "num_subsample": n // b,
                    "num_repeat": d,
                    "exp_dir": exp_dir,
                    "banned_indices_toggle": True,
                    "delays_method_channel": "nr",
                }
                get_qary_indices(params)

                params.update({
                    "sampling_function": scorer.compute_scores,
                    "num_subsample": n // b,
                })
                compute_gfast_samples(params)

                # Run GFast
                params.update({
                    "noise_sd": 0.01,
                    "hyperparam": True,
                    "hyperparam_range": [0, 1, 0.02],
                })
                run_gfast(params)

                # Test on standardized test set
                gwht = np.load(f'{exp_dir}gfast_transform.pickle', allow_pickle=True)
                nmse = test_nmse(test_samples, test_samples_y, gwht, q, n, exp_dir, banned_indices)
                print(f"-----> Test NMSE: {nmse}")
                np.save(f'{exp_dir}nmse.npy', nmse)
            else:
                nmse = np.load(f'{exp_dir}nmse.npy')

            data.append({
                'Method': 'GFast',
                'Samples': calculate_samples(qs, n//b, b, d),
                'b': b,
                'd': d,
                'nmse': nmse
            })


    
    print('Running q-SFT')
    for b in bs:
        for d in ds:

            # Generate indices for sampling
            exp_dir = f"../tabular_results/{dataset}_qsft/q{q}_n{n}_b{b}_d{d}/"
            if os.path.exists(f'{exp_dir}nmse.npy'):
                # Test on standardized test set
                gwht = np.load(f'{exp_dir}gfast_transform.pickle', allow_pickle=True)
                nmse = test_nmse(test_samples, test_samples_y, gwht, q, n, exp_dir, {})
                print(f"-----> Test NMSE: {nmse}")
                np.save(f'{exp_dir}nmse.npy', nmse)

            if not os.path.exists(f'{exp_dir}nmse.npy'):
                params = {
                    "qs": qs,
                    "n": n,
                    "b": b,
                    "num_subsample": n // b,
                    "num_repeat": d,
                    "exp_dir": exp_dir,
                    "banned_indices_toggle": False,
                    "delays_method_channel": "nr",
                }
                get_qary_indices(params)

                params.update({
                    "sampling_function": scorer.compute_scores,
                    "num_subsample": n // b,
                })
                compute_gfast_samples(params)

                # Run GFast
                params.update({
                    "noise_sd": 0.01,
                    "hyperparam": True,
                    "hyperparam_range": [0, 1, 0.02],
                })
                run_gfast(params)

                # Test on standardized test set
                gwht = np.load(f'{exp_dir}gfast_transform.pickle', allow_pickle=True)
                nmse = test_nmse(test_samples, test_samples_y, gwht, q, n, exp_dir, {})
                print(f"-----> Test NMSE: {nmse}")
                np.save(f'{exp_dir}nmse.npy', nmse)
            else:
                nmse = np.load(f'{exp_dir}nmse.npy')


            data.append({
                'Method': 'q-SFT',
                'Samples': calculate_samples_qary(exp_dir, n, b, d, scorer),
                'b': b,
                'd': d,
                'nmse': nmse
            })


    df = pd.DataFrame(data)
    df.to_csv(f'../tabular_results/q{q}_n{n}.csv')