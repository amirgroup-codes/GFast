'''
Methods for the query generator: specifically, to

1. generate sparsity coefficients b and subsampling matrices M
2. get the indices of a signal subsample
3. compute a subsampled and delayed Fourier transform.
'''
import time
import numpy as np
from gfast.utils import fwht, gwht, bin_to_dec, binary_ints, qary_ints, get_qs, banned_random


# def get_Ms_simple(n, b, q, num_to_get=None):
#     '''
#     Sets Ms[0] = [I 0 ...], Ms[1] = [0 I ...], Ms[2] = [0 0 I 0 ...] and so forth. See get_Ms for full signature.
#     '''
#     Ms = []
#     for i in range(num_to_get - 1, -1, -1):
#         M = np.zeros((n, b), dtype=np.int32)
#         M[(b * i) : (b * (i + 1)), :] = np.eye(b) 
#         Ms.append(M)
#     return Ms

def get_Ms_simple(n, b, q, autocomplete = False, num_to_get=None):
    '''
    Sets Ms[0] = [I 0 ...], Ms[1] = [0 I ...], Ms[2] = [0 0 I 0 ...] and so forth. See get_Ms for full signature.
    '''
    Ms = []
    for i in range(num_to_get - 1, -1, -1):
        M = np.zeros((n, b), dtype=np.int32)
        M[(b * i) : (b * (i + 1)), :] = np.eye(b) 
        Ms.append(M)
    if autocomplete:
        last_b = n - b * num_to_get
        if last_b > 0:
            M = np.zeros((n, last_b), dtype=np.int32)
            M[n - last_b: n, 0:last_b] = np.eye(last_b)
            Ms.append(M)
    return Ms


def get_Ms_complex(n, b, q, num_to_get=None):
    """
    Generate M uniformly at random.
    """
    Ms = []
    #TODO Prevent duplicate M (Not a problem for large n, m )
    for i in range(num_to_get):
        M = np.random.randint(q, size=(n, b))
        M = M 
        Ms.append(M)
    M1 = np.ones((n, b), dtype=int)
    M2 = np.full((n, b), 2, dtype=int)
    # Ms[0] = M1
    # Ms[1] = M2
    return Ms


def get_Ms_complex_banned(b, qs, num_to_get=None):
    '''
    gives all m's not including banned indices
    '''
    Ms = []
    for _ in range(num_to_get):
        M = banned_random(b, qs)
        Ms.append(M)
    return Ms


def get_Ms(n, b, q, qs=None, num_to_get=None, method="simple", autocomplete=False):
    '''
    Gets subsampling matrices for different sparsity levels.

    Arguments
    ---------
    n : int
    log_q of the signal length (number of inputs to function).

    b : int
    subsampling dimension.

    num_to_get : int
    The number of M matrices to return.

    method : str
    The method to use. All methods referenced must use this signature (minus "method".)

    Returns
    -------
    Ms : list of numpy.ndarrays, shape (n, b)
    The list of subsampling matrices.
    '''
    if num_to_get is None:
        num_to_get = max(n // b, 3)

    if method == "simple" and num_to_get > n // b and not autocomplete:
        raise ValueError("When query_method is 'simple', the number of M matrices to return cannot be larger than n // b")
    # if qs is not None and method == 'complex':
    #     return get_Ms_complex_banned(b, qs, num_to_get=num_to_get)
    if method == "simple":
        return get_Ms_simple(n, b, q, autocomplete=autocomplete, num_to_get=num_to_get)
    elif method == "complex":
        return get_Ms_complex(n, b, q, qs, autocomplete=autocomplete,num_to_get=num_to_get)
    elif method == "permuted":
        return get_Ms_permuted(n, b, q, qs, autocomplete=autocomplete, num_to_get=num_to_get)


def get_D_identity(n, **kwargs):
    int_delays = np.zeros(n, )
    int_delays = np.vstack((int_delays, np.eye(n)))
    return int_delays.astype(int)


def get_D_random(n, **kwargs):
    '''
    Gets a random delays matrix of dimension (num_delays, n). See get_D for full signature.
    '''
    q=kwargs.get("q")
    num_delays = kwargs.get("num_delays")
    return np.random.choice(q, (num_delays, n))



def get_D_nr(n, D_source, **kwargs):
    '''
    Get a repetition code based (NR) delays matrix. See get_D for full signature.
    '''
    banned_indices_toggle = kwargs.get('banned_indices_toggle')
    if banned_indices_toggle:
        return get_D_nr_banned(n, D_source, **kwargs)
    num_repeat = kwargs.get("num_repeat")
    q = kwargs.get("q")
    random_offsets = get_D_random(n, q=q, num_delays=num_repeat)
    D = []
    for row in random_offsets:
        modulated_offsets = (row - D_source) % q
        D.append(modulated_offsets)
    return D


def get_D_nr_banned(n , D_source, **kwargs):
    '''
    Not sure how useful this is, again we must assume that P (#rows in D) is num_repeat * (n + 1)
    '''
    banned_indices = kwargs.get("banned_indices")
    num_repeat = kwargs.get("num_repeat")
    q_max = kwargs.get("q")
    qs = get_qs(q_max, n, banned_indices=banned_indices)
    random_offsets = get_D_random(n, q=q_max, num_delays=num_repeat, banned_indices_toggle=True, banned_indices=banned_indices)
    D = []
    for i, row in enumerate(random_offsets):
        modulated_offsets = np.zeros((n+1, n), dtype=int)
        for j, D_row in enumerate(D_source):
            if j != 0:
                modulated_offsets_row = (row - D_row) % qs[j-1]
            else:
                modulated_offsets_row = row
            
            modulated_offsets[j,:] = modulated_offsets_row
        D.append(modulated_offsets)
    return D


def get_D_channel_coded(n, D, **kwargs):
    raise NotImplementedError("One day this might be implemented")


def get_D_channel_identity(n, D, **kwargs):
    q = kwargs.get("q")
    return [D % q]


def get_D(n, **kwargs):
    '''
    Delay generator: gets a delays matrix.

    Arguments
    ---------
    n : int
    number of bits: log2 of the signal length.

    Returns
    -------
    D : numpy.ndarray of binary ints, dimension (num_delays, n).
    The delays matrix; if num_delays is not specified in kwargs, see the relevant sub-function for a default.
    '''
    delays_method_source = kwargs.get("delays_method_source", "random")
    D = {
        "random": get_D_random,
        "identity": get_D_identity,
    }.get(delays_method_source)(n, **kwargs)
    delays_method_channel = kwargs.get("delays_method_channel", "identity")
    D = {
            "nr": get_D_nr,
            "coded": get_D_channel_coded,
            "identity": get_D_channel_identity
    }.get(delays_method_channel)(n, D, **kwargs)
    return D


def subsample_indices(M, d):
    '''
    Query generator: creates indices for signal subsamples.

    Arguments
    ---------
    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.

    d : numpy.ndarray, shape (n,)
    The subsampling offset; takes on binary values.

    Returns
    -------
    indices : numpy.ndarray, shape (B,)
    The (decimal) subsample indices. Mostly for debugging purposes.
    '''
    L = binary_ints(M.shape[1])
    inds_binary = np.mod(np.dot(M, L).T + d, 2).T
    return bin_to_dec(inds_binary)

def get_Ms_permuted(n, b, q, qs, autocomplete=False, num_to_get=None):
    simple_Ms = get_Ms_simple(n, b, q, autocomplete=autocomplete, num_to_get=num_to_get)
    descending_indices = np.argsort(qs)[::-1]
    current_index = 0
    Ms_permuted = []
    for M in simple_Ms:
        b_current = M.shape[1]
        selected_indices = descending_indices[current_index:current_index + b_current]
        current_index += b_current
        new_M = np.zeros((n, b_current), dtype=int)
        for j, idx in enumerate(selected_indices):
            new_M[idx, j] = 1
        Ms_permuted.append(new_M)
    return Ms_permuted

def compute_delayed_gwht(signal, M, D, q):
    """
    Computes the Fourier transform of the delayed signal for some M and for each row in the delay matrix D
    """
    b = M.shape[1]
    L = np.array(qary_ints(b, q))  # List of all length b qary vectors
    base_inds = [(M @ L + np.outer(d, np.ones(q ** b, dtype=int))) % q for d in D]
    used_inds = np.swapaxes(np.array(base_inds), 0, 1)
    used_inds = np.reshape(used_inds, (used_inds.shape[0], -1))
    samples_to_transform = signal.get_time_domain(base_inds)
    transform = np.array([gwht(row, q, b) for row in samples_to_transform])
    return transform, used_inds


def get_Ms_and_Ds(n, q, qs=None, **kwargs):
    """
    Based on the parameters provided in kwargs, generates Ms and Ds.
    """
    timing_verbose = kwargs.get("timing_verbose", False)
    if timing_verbose:
        start_time = time.time()
    if 'train_samples' in kwargs:
        query_method = kwargs.get("train_samples")
    else:
        query_method = kwargs.get("query_method")
    autocomplete = kwargs.get("autocomplete", False)
    b = kwargs.get("b")
    num_subsample = kwargs.get("num_subsample")
    Ms = get_Ms(n, b, q, qs=qs, method=query_method, num_to_get=num_subsample, autocomplete=autocomplete)
    if timing_verbose:
        print(f"M Generation:{time.time() - start_time}")
    Ds = []
    if timing_verbose:
        start_time = time.time()
    D = get_D(n, q=q, **kwargs)
    if timing_verbose:
        print(f"D Generation:{time.time() - start_time}")
    for M in Ms:
        Ds.append(D)
    return Ms, Ds


def compute_delayed_wht(signal, M, D):
    '''
    Creates random delays, subsamples according to M and the random delays,
    and returns the subsample WHT along with the delays.

    Arguments
    ---------
    signal : Signal object
    The signal to subsample, delay, and compute the WHT of.

    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.

    num_delays : int
    The number of delays to apply; or, the number of rows in the delays matrix.

    force_identity_like : boolean
    Whether to make D = [0; I] like in the noiseless case; for debugging.
    '''
    inds = np.array([subsample_indices(M, d) for d in D])
    used_inds = set(np.unique(inds))
    samples_to_transform = signal.signal_t[np.array([subsample_indices(M, d) for d in D])] # subsample to allow small WHTs
    return np.array([fwht(row) for row in samples_to_transform]), used_inds # compute the small WHTs

