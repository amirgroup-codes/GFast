import numpy as np
from gfast.utils import igwht_tensored, random_signal_strength_model, qary_vec_to_dec, sort_qary_vecs, dec_to_qary_vec, itft_tensored, qary_vector_banned, decimal_banned, get_qs
from gfast.input_signal import Signal
from gfast.input_signal_subsampled import SubsampledSignal
from multiprocess import Pool
import time
from itertools import product
np.random.seed(42)

def banned_signal_w(n, q, sparsity, a_min, a_max, noise_sd = 0, full = True, max_weight = None, banned_indices = {}):
    qs = get_qs(q, n, banned_indices=banned_indices)
    N = np.prod(qs)
    if max_weight == n:
        locq = sort_qary_vecs(np.array([np.random.randint(0, value) for value in qs])).T


def generate_banned_signal_w(n, q, sparsity, a_min, a_max, noise_sd = 0, full = True, max_weight = None, banned_indices = {}):
    '''
    Generates sparse fourier transform of a signal with banned indices
    '''
    qs = get_qs(q, n, banned_indices=banned_indices)
    N = np.prod(qs)
    max_weight = n if max_weight is None else max_weight
    start = time.time()
    if max_weight == n:
        result = np.empty((n, sparsity), dtype=int)
        # Fill each column with random integers using the corresponding limit
        for i, q in enumerate(qs):
            result[i] = np.random.randint(0, q, size=sparsity)
        locq = sort_qary_vecs(result.T).T
    else:
        non_zero_idx_pos = np.random.choice(a=n, size=(sparsity, max_weight))
        locq = np.zeros((n, sparsity), dtype=int)
        for i in range(sparsity):
            non_zero_idx_vals = np.array([np.random.randint(qs[pos]) for pos in non_zero_idx_pos[i, :]])
            locq[non_zero_idx_pos[i, :], i] = non_zero_idx_vals
        locq = sort_qary_vecs(locq.T).T
    indices = np.array([decimal_banned(loc, qs) for loc in locq.T])
    strengths = random_signal_strength_model(sparsity, a_min, a_max)
    if full:
        wht = np.zeros(N, dtype=complex)
        for l, s in zip(indices, strengths):
            wht[l] = s
        wht += np.random.normal(0, noise_sd, size=(N, 2)).view(complex).reshape(N)
        wht = np.reshape(wht, qs)
        return wht, locq, strengths
    else:
        signal_w = dict(zip(list(map(tuple, locq.T)), strengths))
        return signal_w, locq, strengths


def generate_signal_w(n, q, sparsity, a_min, a_max, noise_sd=0, full=True, max_weight=None):
    """
    Generates a sparse fourier transform
    """
    max_weight = n if max_weight is None else max_weight
    N = (q ** n)

    if max_weight == n:
        locq = sort_qary_vecs(np.random.randint(q, size=(n, sparsity)).T).T
    else:
        non_zero_idx_vals = np.random.randint(q-1, size=(max_weight, sparsity))+1
        non_zero_idx_pos = np.random.choice(a=n, size=(sparsity, max_weight))

        locq = np.zeros((n, sparsity), dtype=int)
        for i in range(sparsity):
            locq[non_zero_idx_pos[i, :], i] = non_zero_idx_vals[:, i]
        locq = sort_qary_vecs(locq.T).T

    loc = qary_vec_to_dec(locq, q)
    strengths = random_signal_strength_model(sparsity, a_min, a_max)

    if full:
        wht = np.zeros((N,), dtype=complex)
        for l, s in zip(loc, strengths):
            wht[l] = s
        signal_w = wht + np.random.normal(0, noise_sd, size=(N, 2)).view(complex).reshape(N)
        return np.reshape(signal_w, [q] * n), locq, strengths
    else:
        signal_w = dict(zip(list(map(tuple, locq.T)), strengths))
        return signal_w, locq, strengths


def get_random_signal(n, q, noise_sd, sparsity, a_min, a_max, banned_indices = {}, banned_indices_toggle = False):
    """
    Computes a full random time-domain signal, which is sparse in the frequency domain. This function is only suitable for
    small n since for large n, storing all q^n symbols is not tractable.
    """
    if banned_indices_toggle:
        signal_w, locq, strengths = generate_banned_signal_w(n, q, sparsity, a_min, a_max, noise_sd, full=True, banned_indices = banned_indices)
        qs = get_qs(q, n, banned_indices=banned_indices)
        signal_t = itft_tensored(signal_w, q, n, qs=qs)
    else:
        signal_w, locq, strengths = generate_signal_w(n, q, sparsity, a_min, a_max, full=True)
        signal_t = igwht_tensored(signal_w, q, n)
    signal_params = {
        "n": n,
        "q": q,
        "noise_sd": noise_sd,
        "signal_t": signal_t,
        "signal_w": signal_w
    }
    return SyntheticSignal(locq, strengths, **signal_params)


class SyntheticSignal(Signal):
    """
    This is essentially just a signal object, except the strengths and locations of the non-zero indicies are known, and
    included as attributes
    """
    def __init__(self, locq, strengths, **kwargs):
        super().__init__(**kwargs)
        self.locq = locq
        self.strengths = strengths


def get_random_subsampled_signal(n, q, noise_sd, sparsity, a_min, a_max, query_args, max_weight=None, banned_indices_toggle = False, banned_indices = {}):
    """
    Similar to get_random_signal, but instead of returning a SyntheticSignal object, it returns a SyntheticSubsampledSignal
    object. The advantage of this is that a subsampled signal does not compute the time domain signal on creation, but
    instead, creates it on the fly. This should be used (1) when n is large or (2) when sampling is expensive.
    """
    start_time = time.time()
    if banned_indices_toggle:
        signal_w, locq, strengths = generate_banned_signal_w(n, q, sparsity, a_min, a_max, noise_sd, full=False, max_weight=max_weight, banned_indices=banned_indices)
    else:
        signal_w, locq, strengths = generate_signal_w(n, q, sparsity, a_min, a_max, noise_sd, full=False, max_weight=max_weight)
    signal_params = {
        "n": n,
        "q": q,
        "query_args": query_args,
        "banned_indices_toggle": banned_indices_toggle,
        "banned_indices": banned_indices
    }
    return SyntheticSubsampledSignal(signal_w=signal_w, locq=locq, strengths=strengths,
                                     noise_sd=noise_sd, **signal_params)


class SyntheticSubsampledSignal(SubsampledSignal):
    """
    This is a Subsampled signal object, except it implements the unimplemented 'subsample' function.
    """
    def __init__(self, **kwargs):
        self.q = kwargs["q"]
        self.n = kwargs["n"]
        self.locq = kwargs["locq"]
        self.noise_sd = kwargs["noise_sd"]
        freq_normalized = 2j * np.pi * kwargs["locq"] / kwargs["q"]
        self.strengths = kwargs["strengths"]
        self.banned_indices_toggle = kwargs.get('banned_indices_toggle')
        self.banned_indices = kwargs.get('banned_indices')
        if self.banned_indices_toggle:
            qs = get_qs(self.q, self.n, banned_indices=self.banned_indices)
            freq_normalized = 2j * np.pi * kwargs["locq"] / qs[:, np.newaxis]

        def sampling_function(query_batch):
            if self.banned_indices_toggle:
                query_indices_qary_batch = np.array([qary_vector_banned(x, qs) for x in query_batch])
            else:
                query_indices_qary_batch = np.array(dec_to_qary_vec(query_batch, self.q, self.n)).T
            return np.exp(query_indices_qary_batch @ freq_normalized) @ self.strengths

        self.sampling_function = sampling_function

        super().__init__(**kwargs)


    def subsample(self, query_indices):
        """
        Computes the signal/function values at the queried indicies on the fly
        """
        batch_size = 10000
        res = []
        query_indices_batches = np.array_split(query_indices, len(query_indices)//batch_size + 1)
        with Pool() as pool:
            for new_res in pool.imap(self.sampling_function, query_indices_batches):
                res = np.concatenate((res, new_res))
        return res


    def get_MDU(self, ret_num_subsample, ret_num_repeat, b, trans_times=False):
        """
        wraps get_MDU method from SubsampledSignal to add synthetic noise
        """
        np.random.seed(42) # Defined twice so multiple calls of the function will use the same noise
        mdu = super().get_MDU(ret_num_subsample, ret_num_repeat, b, trans_times)
        for i in range(len(mdu[2])):
            for j in range(len(mdu[2][i])):
                
                if self.banned_indices_toggle:
                    nu = self.noise_sd / np.sqrt(2 * np.prod(self.qs_subset[ret_num_subsample - i - 1]))
                    # Account for zero-padding
                    size = mdu[2][i][j][:, -np.prod(self.qs_subset[ret_num_subsample - i - 1]):].shape
                    mdu[2][i][j][:, -np.prod(self.qs_subset[ret_num_subsample - i - 1]):] += np.random.normal(0, nu, size=size + (2,)).view(complex).reshape(size)
                else:
                    size = np.array(mdu[2][i][j]).shape
                    nu = self.noise_sd / np.sqrt(2 * self.q ** b)
                    mdu[2][i][j] += np.random.normal(0, nu, size=size + (2,)).view(complex).reshape(size)
        return mdu