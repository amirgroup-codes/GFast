import numpy as np
import os
import sys
sys.path.append('../')
import pandas as pd
import argparse
from qsgft.utils import load_data

# signal_t = load_data('/usr/scratch/dtsui/FinalizedCodes/qsgft/gfp_results/q20_n10_nso/delta0_b1_d1/test/signal_t_mean.pickle')
# # print(signal_t)
# query_indices = load_data('/usr/scratch/dtsui/FinalizedCodes/qsgft/gfp_results/q20_n3_nso/delta0_b2_d40/test/signal_t_queryindices.pickle')
# qary_indices = load_data('/usr/scratch/dtsui/FinalizedCodes/qsgft/gfp_results/q20_n3_nso/delta0_b2_d40/test/signal_t_query_qaryindices.pickle')
# #values = load_data("/usr/scratch/dtsui/FinalizedCodes/qsgft/gfp_results/q20_n6_nso/delta0_b2_d40/test/signal_t_mean.pickle")
# print(query_indices)
# print(qary_indices)
# # print(values)
one_hot_regular = load_data('/usr/scratch/dtsui/FinalizedCodes/qsgft/gfp_results/q20_n3_nso/delta0_b2_d40/test/seqs_regular.pickle')
print("regular sequences", one_hot_regular)
one_hot_banned = load_data('/usr/scratch/dtsui/FinalizedCodes/qsgft/gfp_results/q20_n3_nso/delta1_b2_d40/test/seqs_banned.pickle')
print("banned sequences", one_hot_banned)

print("First sequence banned:", one_hot_banned[0])
print("First sequence regular:", one_hot_regular[0])
print(one_hot_banned[0] == one_hot_regular[0])

print("Type of one_hot_banned:", type(one_hot_banned))
print("Type of one_hot_regular:", type(one_hot_regular))

print("Type of one_hot_banned[0]:", type(one_hot_banned[0]))
print("Type of one_hot_regular[0]:", type(one_hot_regular[0]))

# If they're numpy arrays, check shape:
if isinstance(one_hot_banned[0], np.ndarray):
    print("Shape of one_hot_banned[0]:", one_hot_banned[0].shape)
    print("Shape of one_hot_regular[0]:", one_hot_regular[0].shape)

for i, (b_seq, r_seq) in enumerate(zip(one_hot_banned, one_hot_regular)):
    if b_seq != r_seq:
        print(f"Difference found in outer index {i}")

b_np = np.array(one_hot_banned, dtype=object)  # or dtype=str, ...
r_np = np.array(one_hot_regular, dtype=object)
print("Shapes:", b_np.shape, r_np.shape)
print("array_equal?", np.array_equal(b_np, r_np))

if isinstance(one_hot_banned[0], (bytes, np.bytes_)):
    print("banned[0] is bytes:", one_hot_banned[0])
if isinstance(one_hot_regular[0], (bytes, np.bytes_)):
    print("regular[0] is bytes:", one_hot_regular[0])

print(repr(one_hot_banned[0]))
print(repr(one_hot_regular[0]))

if not np.array_equal(one_hot_banned, one_hot_regular):
    diff_indices = np.where(one_hot_banned != one_hot_regular)
    print(diff_indices)

b_np = np.array(one_hot_banned)   # might become shape (5, 237) with dtype=str
r_np = np.array(one_hot_regular)

# Now we can do a real elementwise comparison:
diff_rows, diff_cols = np.where(b_np != r_np)
print(diff_rows, diff_cols)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate test indices.")
#     parser.add_argument("--n_samples", type=int, required=True)
#     parser.add_argument("--q", type=int, required=True)
#     parser.add_argument("--n", type=int, required=True)
#     parser.add_argument("--banned_indices_path", type=str, required=True, default=None)


#     args = parser.parse_args()
#     banned_indices_path = args.banned_indices_path
#     q = args.q
#     n = args.n
#     n_samples = args.n_samples
#     if banned_indices_path is not None:
#         banned_indices = np.load(banned_indices_path, allow_pickle=True)
#     else:
#         banned_indices = {}
#     qs = get_banned_indices_from_qs(banned_indices, q=q, n=n)
#     N = np.prod(qs)
#     test_indices = np.random.choice(N, n_samples, replace=False)
#     mixedradix_indices = np.array([qary_vector_banned(i, qs) for i in test_indices])
#     qary_indices = np.array([dec_to_qary_vec(i, q) for i in test_indices])
#     #query the model for each index
    
    