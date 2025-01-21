'''
Utility functions.
'''
import numpy as np
import scipy.fft as fft
#from group_lasso import GroupLasso
from sklearn.linear_model import Ridge
import itertools
import math
import random
import time
from scipy.spatial import ConvexHull
import zlib
import pickle
import json
import matplotlib.pyplot as plt
from itertools import product

"""
GFast functions
"""
def ftft(x, q, n, qs = None):
    '''
    Returns FFT of input signal x with alphabet size at each site stored in qs.
    '''
    if qs is None:
        qs = get_qs(q, n)
    signal_length = np.prod(qs)
    x_tensor = np.reshape(x, qs)
    x_tf = fft.fftn(x_tensor) / signal_length
    x_tf = np.reshape(x_tf, [signal_length])
    return x_tf


def iftft(x, q, n, qs = None):
    '''
    Returns IFFT of input signal x with alphabet size at each site stored in qs.
    '''
    if qs is None:
        qs = get_qs(q, n)
    signal_length = np.prod(qs)
    x_tensor = np.reshape(x, qs)
    x_tf = fft.ifftn(x_tensor) * signal_length
    x_tf = np.reshape(x_tf, [signal_length]) 
    return x_tf


def itft_tensored(x, q, n, qs = None):
    '''
    Returns IFFT of tensor as tensor with alphabet size at each site stored in qs.
    '''
    if qs is None:
        qs = get_qs(q,n)
    signal_length = np.prod(qs)
    #return fft.ifftn(x)
    return fft.ifftn(x) * signal_length


def qary_ints_banned(qs, dtype=int):
    '''
    Returns a vector with all of the possible q-ary vectors with alphabets 'qs' 
    then this function returns the same thing as the function 'qary_ints' found in utils.py. 
    This is the matrix L: dimensions are m x (q**m - np.prod(qs))
    '''
    ranges = [np.arange(q) for q in qs]
    return np.array(list(product(*ranges)), dtype=dtype).T


def get_qs(q_max, n, banned_indices={}):
    '''
    Returns the alphabet size (qs) for each site (n) in the sequence. 
    Returns length n vector of only qs if banned_indices arg is left blank.
    '''
    qs = np.repeat(q_max, n)
    for key, value in banned_indices.items():
        qs[key] -= len(value) 
    return qs


def qary_single_to_dec(x, q):
    n = x.size
    decimal_value = np.sum(x * (q ** np.arange(n - 1, -1, -1)))
    return int(decimal_value)


def get_signature(q, n, D, k, banned_indices = {}):
    '''
    Generates the signature to be peeled from a bin with a certain size of banned indices.
    Returns a numpy array length P
    '''
    qs = get_qs(q, n, banned_indices)
    omegas = np.exp(2j * np.pi/qs)
    signature_banned = np.zeros(np.shape(D)[0], dtype=complex)
    for i in range(np.shape(D)[0]):
        row = D[i,:]
        dcp = np.multiply(row, np.ravel(k)) % qs #KUNAL; might need to add mod q here
        omega_row = omegas ** dcp
        signature_banned[i] = np.prod(omega_row)
    return signature_banned


# def qary_vector_banned(x, qs):
#     '''
#     Returns the qary vector of input x with alphabet size at each site stored in qs.
#     Equivalent to dec_to_qary_vec(x, q) for constant qs
#     '''
#     result = []
#     for q in reversed(qs):
#         result.append(x % q)
#         x //= q
#     return tuple(reversed(result))


def qary_vector_banned(x, qs):
    """
    Converts an array of numbers x into their mixed-radix representations
    using the bases provided in qs. This version supports large numbers in x.

    Parameters:
    - x: Array-like or scalar of numbers to convert.
    - qs: List of integers representing the alphabet size at each site.

    Returns:
    - A NumPy array of shape (len(x), len(qs)) representing the mixed-radix vectors.
    """
    x = np.asarray(x, dtype=object)  # Ensure large number support
    result = []

    # Compute mixed-radix representation
    for q in reversed(qs):
        remainder = x % q  # Get the remainder for the current base
        result.append(remainder)
        x //= q  # Update x for the next base

    # Reverse and stack results to get the final mixed-radix representation
    return np.array(result[::-1]).T  # Transpose to align with the input size

def calculate_index(values, dims):
        index = 0
        multiplier = 1
        for i in range(len(dims)-1, -1, -1):
            index += int(values[i]) * multiplier
            multiplier *= int(dims[i])
        return index


def decimal_banned(x, qs):
    '''
    Returns the decimal value of input qary vector x with alphabet size at each site stored in qs.
    Equivalent to qary_vec_to_dec(x, q) for constant qs
    ''' 
    return calculate_index(x, qs)


def find_matrix_indices(x, qs):
    indices = [calculate_index(col, qs) for col in x.T]
    return indices



def banned_random(b, qs):
    '''
    Gives a random matrix not including the values in banned indices
    '''
    M = np.empty((len(qs), b), dtype=int)
    for i, q in enumerate(qs):
        row = np.random.choice(q+1, size=(b))
        M[i] = row
    print(qs, M)
    return M


"""
Base q-SFT functions
"""
def fwht(x):
    """Recursive implementation of the 1D Cooley-Tukey FFT"""
    # x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N == 1:
        return x
    else:
        X_even = fwht(x[0:(N//2)])
        X_odd = fwht(x[(N//2):])
        return np.concatenate([(X_even + X_odd),
                               (X_even - X_odd)])


def gwht(x,q,n):
    """Computes the GWHT of an input signal with forward scaling"""
    x_tensor = np.reshape(x, [q] * n)
    x_tf = fft.fftn(x_tensor) / (q ** n)
    #x_tf = fft.fftn(x_tensor)
    x_tf = np.reshape(x_tf, [q ** n])
    return x_tf

def gwht_tensored(x,q,n):
    """Computes the GWHT of an input signal with forward scaling"""
    x_tf = fft.fftn(x) / (q ** n)
    return x_tf

def igwht(x,q,n):
    """Computes the IGWHT of an input signal with forward scaling"""
    x_tensor = np.reshape(x, [q] * n)
    x_tf = fft.ifftn(x_tensor) * (q ** n)
    x_tf = np.reshape(x_tf, [q ** n])
    return x_tf

def igwht_tensored(x,q,n):
    """Computes the IGWHT of an input signal with forward scaling"""
    x_tf = fft.ifftn(x) * (q ** n)
    return x_tf

def bin_to_dec(x):
    n = len(x)
    c = 2**(np.arange(n)[::-1])
    return c.dot(x).astype(int)


def nth_roots_unity(n):
    return np.exp(-2j * np.pi / n * np.arange(n))


def near_nth_roots(ratios, q, eps):
    in_set = np.zeros(ratios.shape, dtype=bool)
    omega = nth_roots_unity(q)
    for i in range(q):
        in_set = in_set | (np.square(np.abs(ratios - omega[i])) < eps)
    is_singleton = in_set.all()
    return is_singleton


def qary_vec_to_dec(x, q):
    n = x.shape[0]
    return np.array([q ** (n - (i + 1)) for i in range(n)], dtype=object).reshape(-1) @ np.array(x,  dtype=object)


def dec_to_qary_vec(x, q, n):
    qary_vec = []
    for i in range(n):
        qary_vec.append(np.array([a // (q ** (n - (i + 1))) for a in x], dtype=object))
        x = x - (q ** (n-(i + 1))) * qary_vec[i]
    return np.array(qary_vec, dtype=int)


def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)


def binary_ints(m):
    '''
    Returns a matrix where row 'i' is dec_to_bin(i, m), for i from 0 to 2 ** m - 1.
    From https://stackoverflow.com/questions/28111051/create-a-matrix-of-binary-representation-of-numbers-in-python.
    '''
    a = np.arange(2 ** m, dtype=int)[np.newaxis,:]
    b = np.arange(m, dtype=int)[::-1,np.newaxis]
    return np.array(a & 2**b > 0, dtype=int)

def angle_q(x,q):
    return (((np.angle(x) % (2*np.pi) // (np.pi/q)) + 1) // 2) % q # Can be made much faster

# def angle_q(x, q):
#     return np.floor(np.angle(x * np.exp(1j * np.pi/q)) * q/(2 * np.pi))

def qary_ints(m, q, dtype=int):
    return np.array(list(itertools.product(np.arange(q), repeat=m)), dtype=dtype).T

def comb(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

def qary_ints_low_order(m, q, order):
    num_of_ks = np.sum([comb(m, o) * ((q-1) ** o) for o in range(order + 1)])
    K = np.zeros((num_of_ks, m))
    counter = 0
    for o in range(order + 1):
        positions = itertools.combinations(np.arange(m), o)
        for pos in positions:
            K[counter:counter+((q-1) ** o), pos] = np.array(list(itertools.product(1 + np.arange(q-1), repeat=o)))
            counter += ((q-1) ** o)
    return K.T

def base_ints(q, m):
    '''
    Returns a matrix where row 'i' is the base-q representation of i, for i from 0 to q ** m - 1.
    Covers the functionality of binary_ints when n = 2, but binary_ints is faster for that case.
    '''
    get_row = lambda i: np.array([int(j) for j in np.base_repr(i, base=q).zfill(m)])
    return np.vstack((get_row(i) for i in range(q ** m)))

def polymod(p1, p2, q, m):
    '''
    Computes p1 modulo p2, and takes the coefficients modulo q.
    '''
    p1 = np.trim_zeros(p1, trim='f')
    p2 = np.trim_zeros(p2, trim='f')
    while len(p1) >= len(p2) and len(p1) > 0:
        p1 -= p1[0] // p2[0] * np.pad(p2, (0, len(p1) - len(p2)))
        p1 = np.trim_zeros(p1, trim='f')
    return np.pad(np.mod(p1, q), (m + 1 - len(p1), 0))

def rref(A, b, q):
    '''
    Row reduction, to easily solve finite field systems.
    '''
    raise NotImplementedError()

def sign(x):
    '''
    Replacement for np.sign that matches the convention (footnote 2 on page 11).
    '''
    return (1 - np.sign(x)) // 2

def flip(x):
    '''
    Flip all bits in the binary array x.
    '''
    return np.bitwise_xor(x, 1)

def random_signal_strength_model(sparsity, a, b):
    magnitude = np.random.uniform(a, b, sparsity)
    phase = np.random.uniform(0, 2*np.pi, sparsity)
    return magnitude * np.exp(1j*phase)


def best_convex_underestimator(points):
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    first_point_idx = np.argmin(vertices[:, 0])
    last_point_idx = np.argmax(vertices[:, 0])

    if last_point_idx == vertices.shape[0]:
        return vertices[first_point_idx:]
    if first_point_idx < last_point_idx:
        return vertices[first_point_idx:last_point_idx+1]
    else:
        return np.concatenate((vertices[first_point_idx:], vertices[:last_point_idx+1]))

def sort_qary_vecs(qary_vecs):
    qary_vecs = np.array(qary_vecs)
    idx = np.lexsort(qary_vecs.T[::-1, :])
    return qary_vecs[idx]

def calc_hamming_weight(qary_vecs):
    qary_vecs = np.array(qary_vecs)
    return np.sum(qary_vecs != 0, axis = 1)

def save_data(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL), 9))

def load_data(filename):
    start = time.time()
    with open(filename, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))
    return data

def save_data4(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, protocol=4), 9))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

     
def count_interactions(locations):
    nonzero_counts = {}
    for row in locations:
        nonzero_indices = np.nonzero(row)[0]
        num_nonzero_indices = len(nonzero_indices)
        nonzero_counts[num_nonzero_indices] = nonzero_counts.get(num_nonzero_indices, 0) + 1
    
    for num_nonzero_indices, count in nonzero_counts.items():
        print("There are {} {}-order interactions.".format(count, num_nonzero_indices))
    
    return nonzero_counts

def calculate_fourier_magnitudes(locations, gwht):
    nonzero_counts = count_interactions(locations)
    k_values = sorted(nonzero_counts.keys())
    j = 0 if 0 in k_values else 1
    F_k_values = np.zeros(max(np.max(k_values)+1, len(k_values)))

    for row in locations:
        nonzero_indices = np.nonzero(row)[0]
        num_nonzero_indices = len(nonzero_indices)
        F_k_values[num_nonzero_indices-j] += np.abs(gwht[row])
    
    F_k_values = np.square(F_k_values)
    return dict(zip(k_values, F_k_values))

def plot_interaction_magnitudes(sum_squares, q, n, b, output_folder, args):
    index_counts = list(sum_squares.keys())
    values = list(sum_squares.values())
    plt.figure()
    plt.bar(index_counts, values, align='center', color='limegreen')
    plt.xlabel('$r^{th}$ order interactions')
    plt.ylabel('Magnitude of Fourier coefficients')
    plt.xticks(index_counts)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('q{}_n{}_b{}'.format(q, n, b))

    if hasattr(args, 'param'):
        param_path = args.param.replace('/', '_')
        file_path = 'magnitude_of_interactions_{}.png'.format(param_path)
    else:
        file_path = 'magnitude_of_interactions.png'
    plt.savefig(output_folder / file_path)
    plt.close()

def write_results_to_file(results_file, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, qs):
    results_file.parent.mkdir(parents=True, exist_ok=True)
    max = np.repeat(q, n)
    delta = np.sum(max - qs)
    with open(results_file, "w") as file:
        file.write("q = {}, n = {}, b = {}, noise_sd = {}\n".format(q, n, b, noise_sd))
        file.write("\nTotal samples = {}\n".format(n_used))
        file.write("Total sample ratio = {}\n".format(n_used / q ** n))
        file.write("R^2 = {}\n".format(r2_value))
        file.write("NMSE = {}\n".format(nmse))
        file.write("AVG Hamming Weight of Nonzero Locations = {}\n".format(avg_hamming_weight))
        file.write("Max Hamming Weight of Nonzero Locations = {}\n".format(max_hamming_weight))
        file.write("Alphabets = {}\n".format(qs))
        file.write("Total AAs removed = {}\n".format(delta))

        

def summarize_results(locations, gwht, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, folder, args, qs):

    if gwht:
        sum_squares = calculate_fourier_magnitudes(locations, gwht)
        plot_interaction_magnitudes(sum_squares, q, n, b, folder, args)

    if hasattr(args, 'param'):
        param_path = args.param.replace('/', '_')
        file_path = 'helper_results_{}.txt'.format(param_path)
    else:
        file_path = 'helper_results.txt'

    results_file = folder / file_path
    write_results_to_file(results_file, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, qs)


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False


def process_stdout(captured_output):
    # Iterate over captured output in reverse to find the last occurrence of "NMSE"
    for line in reversed(captured_output):
        if "NMSE" in line:
            if "Min NMSE:" in line:
                nmse_value = line.split("Min NMSE:")[-1].strip()
                return nmse_value
            else:
                print(f"Line with 'NMSE' found but no 'Min NMSE:': {line}")
                raise ValueError("'Min NMSE:' not found in the line containing 'NMSE'")
    print('Captured Output:', captured_output)
    raise ValueError("No 'NMSE' found in the output")
    # if captured_output:
    #     last_line = captured_output[-1]
    #     if "Min NMSE:" in last_line:
    #         nmse_value = last_line.split("Min NMSE:")[-1].strip()
    #         return nmse_value
    #     else:
    #         print('hiiiii', captured_output)
    #         raise ValueError("No 'Min NMSE:' found in the output")
    # else:
    #     raise ValueError("No output captured")
        
    # if result.returncode == 0:
    #     # Split the output into lines and capture the last one
    #     output_lines = result.stdout.strip().splitlines()
    #     if output_lines:
    #         last_line = output_lines[-1]
    #         try:
    #             # Extract the part after "Min NMSE:"
    #             if "Min NMSE:" in last_line:
    #                 nmse_value = last_line.split("Min NMSE:")[-1].strip()
    #             else:
    #                 nmse_value = 1
    #         except Exception as e:
    #             return ValueError(f"Error processing the last line: {e}")
    # else:
    #     return ValueError(f"Error: {result.stderr.strip()}")
    # return nmse_value


def test_nmse(test_set, y, gwht, q, n, exp_dir, banned_indices={}):
    if len(gwht.keys()) > 0:
        batch_size = 10000
        beta_keys = list(gwht.keys())
        beta_values = list(gwht.values())
        y_hat = []
        test_set = test_set.T

        for i in range(0, test_set.shape[0], batch_size):
            sample_batch = test_set[i:i + batch_size, :]
            H = np.empty((np.shape(sample_batch)[0], np.shape(beta_keys)[0]), dtype=complex)
            for i, k in enumerate(np.array(beta_keys)):
                signature = get_signature(q, n, sample_batch, k, banned_indices=banned_indices)
                H[:, i] = signature.T
            y_hat.append(H @ np.array(beta_values))
        y_hat = np.concatenate(y_hat)

        # Subtract sample mean from new test set
        mean = np.load(f'{exp_dir}/train/samples/train_mean.npy')
        y = y - mean

        nmse = np.linalg.norm(y_hat - y) ** 2 / np.linalg.norm(y) ** 2
        return nmse


    
    # if len(beta.keys()) > 0:
    #         test_signal = self.test_signal.signal_t
    #         # print("Test signal", test_signal)
    #         qs = get_qs(self.q, self.n, banned_indices=self.test_signal.banned_indices)
    #         (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
    #         # print("SAMPLES/QS", qs, samples)
    #         batch_size = 10000
    #         beta_keys = list(beta.keys())
    #         beta_values = list(beta.values())
    #         y_hat = []

    #         for i in range(0, len(sample_idx_dec), batch_size):
    #             sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
    #             sample_idx_batch = np.array([qary_vector_banned(x, qs) for x in sample_idx_dec_batch])
    #             H = np.empty((np.shape(sample_idx_batch)[0], np.shape(beta_keys)[0]), dtype=complex)
    #             for i, k in enumerate(np.array(beta_keys)):
    #                 signature = get_signature(self.q, self.n, sample_idx_batch, k, banned_indices=self.test_signal.banned_indices)
    #                 H[:, i] = signature.T
    #             y_hat.append(H @ np.array(beta_values))
    #         y_hat = np.concatenate(y_hat)
    #         nmse = np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
    #         r2 = r2_score(np.real(y_hat), np.real(samples))
    #         # print(f"NMSE: {nmse}, R2: {r2}")
    #         import matplotlib.pyplot as plt
    #         plt.figure()
    #         plt.plot(np.real(y_hat), np.real(samples))
    #         plt.savefig('/usr/scratch/dtsui/FinalizedCodes/gfast/gfp_results/q20_n12_nso_0.012/test.png')
    #         return nmse, r2
    #     else:
    #         return 1, -1