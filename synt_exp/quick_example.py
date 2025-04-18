import numpy as np
import sys
import os
import time
from pathlib import Path
sys.path.append('..')
from gfast.gfast import GFAST
from synt_exp.synt_src.synthetic_signal import get_random_subsampled_signal, generate_banned_signal_w, SyntheticSubsampledSignal   
from gfast.utils import get_qs, get_banned_indices_from_qs

SNR = 100
delta = 0.3

if __name__ == '__main__':
    np.random.seed(int(time.time()))
    #np.random.seed(3)
    q = 4
    n = 8
    banned_indices = {0:[0, 1], 1:[0, 1],4:[0, 1], 6:[1]}
    qs = get_qs(q, n, banned_indices=banned_indices)
    print("QS: ", qs, np.prod(qs), q ** n)
    N = np.prod(qs)
    sparsity = int(N ** delta)
    a_min = 1
    a_max = 1
    b = 3
    noise_sd = 0
    print('noise sd:', noise_sd)
    num_subsample = 2
    num_repeat = 1
    t = n
    delays_method_source = "identity"
    delays_method_channel = "identity"
    banned_indices_toggle = True
    query_args = {
        "query_method": "permuted",
        "num_subsample": num_subsample,
        "delays_method_source": delays_method_source,
        "subsampling_method": "gfast",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "b": b,
        "t": t,
        "banned_indices_toggle": banned_indices_toggle,
        "banned_indices": banned_indices,
        "autocomplete": True
    }
    qsft_args = {
        "num_subsample": num_subsample,
        "num_repeat": num_repeat,
        "reconstruct_method_source": delays_method_source,
        "reconstruct_method_channel": delays_method_channel,
        "b": b,
        "noise_sd": noise_sd
    }
    signal_params = {
        "n": n,
        "q": q,
        "banned_indices_toggle": banned_indices_toggle,
        "query_args": query_args,
        "banned_indices": banned_indices
    }
    start_synt = time.time()
    signal_w, locq, strengths = generate_banned_signal_w(n, q, sparsity, a_min, a_max, noise_sd, full=False, banned_indices=banned_indices, max_weight=t)
    #print("strengths", strengths)
    test_signal = SyntheticSubsampledSignal(signal_w=signal_w, locq=locq, strengths=strengths, noise_sd=noise_sd, **signal_params)
    end_synt = time.time()
    start_transform = time.time()
    print("starting GFAST")
    sft = GFAST(**qsft_args)
    print("finished GFAST")
    result = sft.transform(test_signal, verbosity=0, timing_verbose=False, report=True, sort=True)
    end_transform = time.time()
    #print("Ground truth coefficients: ", test_signal.signal_w)
    gwht = result.get("gwht")
    #print("Recovered coefficients: ", gwht)
    loc = result.get("locations")
    n_used = result.get("n_samples")
    peeled = result.get("locations")
    avg_hamming_weight = result.get("avg_hamming_weight")
    max_hamming_weight = result.get("max_hamming_weight")
    print("found non-zero indices GFAST: ")
    print(peeled)
    print("True non-zero indices: ")
    print(test_signal.locq.T)
    print("Total samples = ", n_used)
    print("Total sample ratio = ", n_used / q ** n)
    signal_w_diff = test_signal.signal_w.copy()
    for key in gwht.keys():
        signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
    print('Num coeffs:', len(gwht), len(test_signal.signal_w))
    print("NMSE GFAST = ", np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(test_signal.signal_w.values())) ** 2))
    print("AVG Hamming Weight of Nonzero Locations = ", avg_hamming_weight)
    print("Max Hamming Weight of Nonzero Locations = ", max_hamming_weight)