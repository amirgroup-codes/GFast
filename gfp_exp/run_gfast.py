import numpy as np
import argparse
from src.helper import Helper
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from gfast.utils import str_to_bool, get_qs, summarize_results
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GFast.")
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--num_subsample", type=int, required=True)
    parser.add_argument("--num_repeat", type=int, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--banned_indices_toggle", type=str, required=True)
    parser.add_argument("--banned_indices_path", type=str, required=False, default=None)
    parser.add_argument("--delays_method_source", type=str, required=False, default="identity")
    parser.add_argument("--delays_method_channel", type=str, required=False, default="identity")
    parser.add_argument("--hyperparam", type=str, required=False, default=False)


    args = parser.parse_args()
    q = args.q
    n = args.n
    b = args.b
    num_subsample = args.num_subsample
    num_repeat = args.num_repeat
    exp_dir = Path(args.exp_dir)
    t = 3
    delays_method_source = args.delays_method_source
    delays_method_channel = args.delays_method_channel
    banned_indices_toggle = str_to_bool(args.banned_indices_toggle)
    if args.banned_indices_path is not None:
        banned_indices = np.load(args.banned_indices_path, allow_pickle=True)
    hyperparam = str_to_bool(args.hyperparam)
    noise_sd = 0.1

    query_args = {
        "query_method": "simple",
        "num_subsample": num_subsample,
        "delays_method_source": delays_method_source,
        "subsampling_method": "gfast",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "b": b,
        "t": t,
        "folder": exp_dir 
    }
    signal_args = {
                    "n":n,
                    "q":q,
                    "noise_sd":noise_sd,
                    "query_args":query_args,
                    "t": t,
                    'banned_indices_toggle': banned_indices_toggle,
                    'banned_indices': banned_indices
                    }
    test_args = {
            "n_samples": 10000
        }
    
    helper = Helper(signal_args=signal_args, methods=["gfast"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b
    test_kwargs = {}


    if hyperparam:
        print('Hyperparameter tuning noise_sd:')
        noise_sd = np.arange(0, 5, 0.1).round(2)
        nmse_entries = []
        r2_entries = []

        for noise in noise_sd:
            signal_args.update({
                "noise_sd": noise
            })
            model_kwargs["noise_sd"] = noise
            model_result = helper.compute_model(method="gfast", model_kwargs=model_kwargs, report=True, verbosity=0)
            test_kwargs["beta"] = model_result.get("gwht")
            nmse, r2 = helper.test_model("gfast", **test_kwargs)
            gwht = model_result.get("gwht")
            locations = model_result.get("locations")
            n_used = model_result.get("n_samples")
            avg_hamming_weight = model_result.get("avg_hamming_weight")
            max_hamming_weight = model_result.get("max_hamming_weight")
            nmse_entries.append(nmse)
            r2_entries.append(r2)
            print(f"noise_sd: {noise} - NMSE: {nmse}, R2: {r2}")

        min_nmse_ind = nmse_entries.index(min(nmse_entries))
        min_nmse = nmse_entries[min_nmse_ind]
        print('----------')
        print(f"b: {b}, d: {num_repeat} - noise_sd: {noise_sd[min_nmse_ind]}, Min NMSE: {min_nmse}")

        # Recompute qsft with the best noise_sd
        signal_args.update({
            "noise_sd": noise_sd[min_nmse_ind]
        })
        model_kwargs["noise_sd"] = noise_sd[min_nmse_ind]
        model_result = helper.compute_model(method="gfast", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse, r2_value = helper.test_model("gfast", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")

        plt.figure()
        plt.title(f'q{q}_n{n}_b{b}')
        plt.plot(noise_sd, nmse_entries[:], marker='o', linestyle='-', color='b')
        plt.scatter(noise_sd[min_nmse_ind], nmse_entries[min_nmse_ind], color='red', marker='x', label='Min NMSE')
        plt.text(noise_sd[min_nmse_ind], nmse_entries[min_nmse_ind], f'noise_sd: {noise_sd[min_nmse_ind]} - Min NMSE: {min_nmse:.2f}', ha='right', va='top')
        plt.xlabel('noise_sd')
        plt.ylabel('NMSE')
        plt.savefig(str(exp_dir) + '/nmse.png')  
        df = pd.DataFrame({'noise_sd': noise_sd, 'nmse': nmse_entries})
        df.to_csv(str(exp_dir) + '/nmse.csv', index=False)

    else:
        print('Running q-sft')
        model_kwargs["noise_sd"] = noise_sd
        model_result = helper.compute_model(method="gfast", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse, r2_value = helper.test_model("gfast", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")
        print('----------')
        print(f"b: {b}, d: {num_repeat} - R^2: {r2_value}, NMSE: {nmse}")

    with open(f'{exp_dir}/gwht.pkl', 'wb') as f:
        pickle.dump(gwht, f)

    if banned_indices_toggle:
        qs = get_qs(q, n, banned_indices)
    else:
        qs = get_qs(q, n, {})
    if type(noise_sd) == np.ndarray:
        noise_sd = noise_sd[min_nmse_ind]
    
    summarize_results(locations, gwht, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, exp_dir, args, qs)

















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
