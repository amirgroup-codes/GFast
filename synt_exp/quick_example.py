import numpy as np
import sys
sys.path.append('..')
from gfast.gfast import GFAST
from gfast.utils import get_qs
import time
from gfast.plot_utils import get_banned_indices_from_qs, get_qs_from_delta, get_qs_from_delta_random, calculate_samples, get_qs_from_delta_sitewise
from synt_exp.synt_src.synthetic_signal import generate_banned_signal_w, SyntheticSubsampledSignal

delta = 0.33
if __name__ == '__main__':
    np.random.seed(int(time.time()))
    q = 3
    n = 8
    b = 2
    qs = get_qs_from_delta(0, q, n)
    print(qs, np.prod(qs))
    banned_indices = get_banned_indices_from_qs(qs, q)
    noise_sd = 2
    a_min = 1
    a_max = 1
    t = n
    sparsity = int(np.prod(qs) ** delta)
    num_subsample = 4
    num_repeat = n
    t = n
    delays_method_source = "identity"
    delays_method_channel = "nso" 
    banned_indices_toggle = True
    query_args = {
        "query_method": "simple",
        "num_subsample": num_subsample,
        "delays_method_source": delays_method_source,
        "subsampling_method": "qsft",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "b": b,
        "t": t,
        "banned_indices_toggle": banned_indices_toggle,
        'banned_indices': banned_indices
    }
    gfast_args = {
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
        'banned_indices_toggle': banned_indices_toggle,
        'query_args': query_args,
        'banned_indices': banned_indices if banned_indices_toggle else {}
    }
    start_synt = time.time()
    signal_w, locq, strengths = generate_banned_signal_w(n, q, sparsity, a_min, a_max, noise_sd, full=False, banned_indices=banned_indices, max_weight=t)
    test_signal = SyntheticSubsampledSignal(signal_w=signal_w, locq=locq, strengths=strengths, noise_sd=noise_sd, **signal_params)
    sft = GFAST(**gfast_args)
    result = sft.transform(test_signal, verbosity=0, timing_verbose=False, report=True, sort=True)
    gwht = result.get("gwht")
    signal_w_diff = test_signal.signal_w.copy()
    for key in gwht.keys():
        signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
    print("Samples Used = ", result.get("n_samples"))
    print('Num coeffs:', len(gwht), len(test_signal.signal_w))
    print("NMSE GFAST = ",
         np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(test_signal.signal_w.values())) ** 2))

    
    