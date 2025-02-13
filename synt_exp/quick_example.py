import numpy as np
import sys
sys.path.append('..')
from gfast.gfast import GFAST
from gfast.utils import get_qs, get_banned_indices_from_qs, get_qs_from_delta, get_qs_from_delta_random, calculate_samples, get_qs_from_delta_sitewise
import time
from synt_exp.synt_src.synthetic_signal import generate_banned_signal_w, SyntheticSubsampledSignal

delta = 0.3
snr = 10
if __name__ == '__main__':
    np.random.seed(int(time.time()))
    #np.random.seed(42)
    q = 4
    n = 6
    b = 2
    qs = np.array([4, 4, 3, 3, 3, 4])
    #qs = get_qs_from_delta(3, q, n)
    print(qs, np.prod(qs))
    banned_indices = get_banned_indices_from_qs(qs, q)
    a_min = 1
    a_max = 1
    t = n
    sparsity = int(np.prod(qs) ** delta)
    print('sparsity:', sparsity)
    #noise_sd = np.sqrt((sparsity * a_max**2) / (10**(snr / 10)))
    noise_sd = 2
    print('noise sd:', noise_sd)
    num_subsample = 3
    num_repeat = n
    t = n
    delays_method_source = "identity"
    delays_method_channel = "nr" 
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

    
    