import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper import Helper
import os
from pathlib import Path
sys.path.append("..")
from gfast.utils import str_to_bool, get_banned_indices_from_qs, load_data, save_data, count_interactions, calculate_fourier_magnitudes, plot_interaction_magnitudes, write_results_to_file, summarize_results, get_random_test_samples, test_nmse
import time
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
from input_signal_subsampled import ScriptExit

def get_qary_indices(params):
    """
    Get q-ary indices to query from GFast

    params: dictionary with the following parameters:
        qs (list): The list of q alphabets.
        n (int): Length of sequence.
        b (int): Subsampling dimension.
        num_subsample (int): The number of subsamples.
        num_repeat (int): The number of repeats.
        exp_dir (str): The experiment directory.
        banned_indices_path (str): The path to the banned indices file.
        banned_indices_toggle (bool): Whether to use banned indices.
        delays_method_channel (str): The delays method channel.
        query_method (str): The query method (simple for GFast).
    """

    # Extract parameters from dictionary
    qs = params["qs"]
    n = params["n"]
    b = params["b"]
    num_subsample = params["num_subsample"]
    num_repeat = params["num_repeat"]
    exp_dir = Path(params["exp_dir"])
    # banned_indices_path = params["banned_indices_path"]
    banned_indices_toggle = params["banned_indices_toggle"]
    delays_method_channel = params["delays_method_channel"]
    if "query_method" not in params:
        params["query_method"] = "simple"
    if "autocomplete" not in params:
        params["autocomplete"] = False
    query_method = params["query_method"]
    autocomplete = params["autocomplete"]

    q = np.max(np.array(qs))

    if delays_method_channel:
        delays_method_channel = params["delays_method_channel"]
    else:
        delays_method_channel = "identity"
    if params["banned_indices_toggle"]:
        banned_indices = get_banned_indices_from_qs(qs, q)
    else:
        banned_indices = {}

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    query_args = {
        "query_method": 'generate_samples',
        "train_samples": query_method,
        "method": "generate_samples",
        "num_subsample": num_subsample,
        "num_repeat": num_repeat,
        "b": b,
        "folder": exp_dir,
        "delays_method_channel": delays_method_channel,
        "autocomplete": autocomplete
    }
    signal_args = {
        "n":n,
        "q":q,
        "query_args":query_args,
        "len_seq":n,
        "banned_indices_toggle": banned_indices_toggle,
        "banned_indices": banned_indices,
        "delays_method_channel": delays_method_channel
        }
    test_args = {
            "n_samples": 10000,
            "method": "generate_samples"
        }
    file_path = os.path.join(exp_dir, 'train', 'samples')
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(exp_dir, 'test')
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    try:
        helper = Helper(signal_args=signal_args, methods=["gfast"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
    except ScriptExit:
        pass
        # Continue with the rest of your main script
    # except RuntimeError as e:
    #     pass
    

def compute_gfast_samples(params):
    """
    Computes samples for a given set of parameters

    params: dictionary with the following parameters:
        qs (list): The list of q alphabets.
        n (int): Length of sequence.
        b (int): Subsampling dimension.
        num_subsample (int): The number of subsamples.
        num_repeat (int): The number of repeats.
        exp_dir (str): The experiment directory.
        banned_indices_path (str): The path to the banned indices file.
        banned_indices_toggle (bool): Whether to use banned indices.
        delays_method_channel (str): The delays method channel.
        query_method (str): The query method (simple for GFast).

        param (str): Experiment name to use.
        sampling_function (function): The sampling function to use. Takes in a numpy array of length (num_samples x n) and returns a 1D numpy array of values (num_samples,).
    """
    # Extract parameters from dictionary
    qs = params["qs"]
    n = params["n"]
    b = params["b"]
    M = params["num_subsample"]
    D = params["num_repeat"]
    exp_dir = params["exp_dir"]
    # banned_indices_path = params["banned_indices_path"]
    banned_indices_toggle = params["banned_indices_toggle"]
    delays_method_channel = params["delays_method_channel"]
    query_method = params["query_method"]
    sampling_function = params["sampling_function"]

    q = np.max(qs)



    """
    Initialize files
    """
    folder_path = os.path.join(exp_dir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, "train"))
        os.makedirs(os.path.join(folder_path, "train", "samples"))
        os.makedirs(os.path.join(folder_path, "test"))
    folder_path = os.path.join(folder_path, "train", "samples")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(exp_dir, "train", "samples_mean")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(exp_dir, "test")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



    """
    Compute samples needed
    """
    # file_path = [1]
    for i in range(M):
        for j in range(D):
            query_indices_file = os.path.join(exp_dir, "train", "samples", "M{}_D{}_queryindices.pickle".format(i, j))
            query_indices = load_data(query_indices_file)
            flag = True

            sample_file = os.path.join(exp_dir, "train", "samples", "M{}_D{}.pickle".format(i, j))
            sample_file_mean = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
            if os.path.isfile(sample_file):
                flag = False

            if flag:
                all_query_indices = np.concatenate(query_indices)

                all_samples = np.zeros((np.shape(all_query_indices)[0], 1))
                block_length = len(query_indices[0])
                samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(1)]
                print('Computing samples for M{}_D{}:'.format(i, j))
                all_samples[:,0] = sampling_function(all_query_indices)
                
                for sample, arr in zip(samples, all_samples.T):
                    for k in range(len(query_indices)):
                        sample[k] = arr[k * block_length: (k+1) * block_length]
                    sample = sample.T
                    save_data(sample, sample_file)
                    save_data(sample, sample_file_mean)



    # Save the empirical mean separately
    folder_path = os.path.join(exp_dir)
    mean_file = os.path.join(exp_dir, "train", "samples", "train_mean.npy") 

    if not os.path.isfile(mean_file):
        all_samples = []
        for i in range(M):
            for j in range(D):
                sample_file = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
                samples = load_data(sample_file)
                samples = np.concatenate(samples)
                all_samples = np.concatenate([all_samples, samples])
        all_samples_mean = np.mean(all_samples)
        np.save(mean_file, all_samples_mean)
    else:
        all_samples_mean = np.load(mean_file)
    
    for i in range(M):
        for j in range(D):
            sample_file_zeromean = os.path.join(exp_dir, "train", "samples", "M{}_D{}.pickle".format(i, j))
            sample_file = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
            samples = load_data(sample_file)
            samples_zeromean = samples - all_samples_mean
            save_data(samples_zeromean, sample_file_zeromean)



    """
    Testing samples to compute NMSE and R^2
    """
    query_indices_file = os.path.join(exp_dir, "test", "signal_t_queryindices.pickle")
    query_indices = load_data(query_indices_file)

    query_qaryindices_file = os.path.join(exp_dir, "test", "signal_t_query_qaryindices.pickle")
    query_qaryindices = load_data(query_qaryindices_file)

    # Loop through all files and check if they exist
    sample_file = os.path.join(exp_dir, "test", "signal_t.pickle")
    sample_file_mean = os.path.join(exp_dir, "test", "signal_t_mean.pickle")
    flag = True

    if os.path.isfile(sample_file):
        flag = False

    if flag:
        all_query_indices = query_indices.T

        all_samples = np.zeros((np.shape(all_query_indices)[0], 1))
        block_length = len(query_indices[0])
        samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(1)]
        print('Computing test samples:')
        all_samples[:,0] = sampling_function(all_query_indices)

        for arr in all_samples.T:
            sample_file = os.path.join(exp_dir, "test", "signal_t.pickle")
            sample_file_mean = os.path.join(exp_dir, "test", "signal_t_mean.pickle")
            samples_dict = dict(zip(query_qaryindices, arr))
            save_data(samples_dict, sample_file)
            save_data(samples_dict, sample_file_mean)

        # Remove empirical mean
        mean_file = os.path.join(exp_dir, "train", "samples", "train_mean.npy")
        all_samples_mean = np.load(mean_file)

        sample_file_mean = os.path.join(exp_dir, "test", "signal_t_mean.pickle")
        sample_file = os.path.join(exp_dir, "test", "signal_t.pickle")
        samples_dict = load_data(sample_file_mean)

        all_values = list(samples_dict.values())
        all_values = np.array(all_values, dtype=complex) - all_samples_mean
        samples_dict = {key: value for key, value in zip(samples_dict.keys(), all_values)}
        save_data(samples_dict, sample_file)


def run_gfast(params):
    """
    Run GFast on a given set of parameters

    params: dictionary with the following parameters:
        qs (list): The list of q alphabets.
        n (int): Length of sequence.
        b (int): Subsampling dimension.
        num_subsample (int): The number of subsamples.
        num_repeat (int): The number of repeats.
        exp_dir (str): The experiment directory.
        banned_indices_path (str): The path to the banned indices file.
        banned_indices_toggle (bool): Whether to use banned indices.
        delays_method_channel (str): The delays method channel.
        query_method (str): The query method (simple for GFast).
        param (str): Experiment name to use.

        noise_sd (float): noise_sd to use if hyperparam is False.
        hyperparam (bool): Whether to use a hyperparameter.
        hyperparam_range (list): The range of hyperparameters search over to find optimal noise_sd is hyperparam is True: [min, max, step].
    """
    start_time = time.time()

    # Extract parameters from dictionary
    qs = params["qs"]
    n = params["n"]
    b = params["b"]
    num_subsample = params["num_subsample"]
    num_repeat = params["num_repeat"]
    exp_dir = params["exp_dir"]
    # banned_indices_path = params["banned_indices_path"]
    banned_indices_toggle = params["banned_indices_toggle"]
    delays_method_source = "identity"
    delays_method_channel = params["delays_method_channel"]
    query_method = params["query_method"]
    noise_sd = params["noise_sd"]
    hyperparam = params["hyperparam"]
    hyperparam_range = params["hyperparam_range"]

    q = np.max(np.array(qs))

    if delays_method_channel:
        delays_method_channel = params["delays_method_channel"]
    else:
        delays_method_channel = "identity"
    if params["banned_indices_toggle"]:
        banned_indices = get_banned_indices_from_qs(qs, q)
    else:
        banned_indices = {}



    """
    Initialization
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    query_args = {
        "query_method": query_method,
        "num_subsample": num_subsample,
        "delays_method_source": delays_method_source,
        "subsampling_method": "gfast",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "b": b,
        "folder": exp_dir 
    }
    signal_args = {
                    "n":n,
                    "q":q,
                    "noise_sd":noise_sd,
                    "query_args":query_args,
                    'banned_indices_toggle': banned_indices_toggle,
                    'banned_indices': banned_indices,
                    }
    test_args = {
            "n_samples": 10000
        }

    helper = Helper(signal_args=signal_args, methods=["gfast"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b



    """
    Recover Fourier coefficients and get summary statistics
    """
    print('----------')
    print("Sampling from model")
    start_time_sampling = time.time()
    helper = Helper(signal_args=signal_args, methods=["gfast"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
    end_time_sampling = time.time()
    elapsed_time_sampling = end_time_sampling - start_time_sampling
    print(f"Sampling time: {elapsed_time_sampling} seconds")

    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b
    test_kwargs = {}
    model_kwargs["n_samples"] = num_subsample * (helper.q ** b) * num_repeat * (helper.n + 1)

    if hyperparam:
        print('Hyperparameter tuning noise_sd:')
        start_time_hyperparam = time.time()
        range_values = [float(x) for x in hyperparam_range]
        noise_sd = np.arange(range_values[0], range_values[1], range_values[2]).round(3)
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

        end_time_hyperparam= time.time()
        elapsed_time_hyperparam = end_time_hyperparam - start_time_hyperparam
        min_nmse_ind = nmse_entries.index(min(nmse_entries))
        min_nmse = nmse_entries[min_nmse_ind]
        print('----------')
        print(f"Hyperparameter tuning time: {elapsed_time_hyperparam} seconds")
        print(f"noise_sd: {noise_sd[min_nmse_ind]} - Min NMSE: {min_nmse}")

        # Recompute gfast with the best noise_sd
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
        noise_sd = noise_sd[min_nmse_ind]

    else:
        print('Running GFast')
        model_kwargs["noise_sd"] = noise_sd
        start_time_gfast = time.time()
        model_result = helper.compute_model(method="gfast", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse, r2_value = helper.test_model("gfast", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")
        end_time_gfast = time.time()
        elapsed_time_gfast = end_time_gfast - start_time_gfast
        print('----------')
        print(f"q-sft time: {elapsed_time_gfast} seconds")
        print(f"R^2 is {r2_value}")
        print(f"NMSE is {nmse}")
        
    with open(str(exp_dir) + "/" + "gfast_transform.pickle", "wb") as pickle_file:
        pickle.dump(gwht, pickle_file)

    if banned_indices_toggle == False:
        # Test on random samples from the subset of qs
        print('Computing NMSE on random samples inside qs')
        banned_indices_qs = get_banned_indices_from_qs(qs, q)
        test_samples_qs = get_random_test_samples(q, n, banned_indices_qs).T
        sampling_function = params["sampling_function"]
        scores_qs = sampling_function(test_samples_qs)
        nmse = test_nmse(test_samples_qs.T, scores_qs, gwht, q, n, exp_dir, {})
        print('NMSE: ', nmse)


    summarize_results(locations, gwht, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, exp_dir, params, qs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time: {elapsed_time} seconds")
    print('----------')
