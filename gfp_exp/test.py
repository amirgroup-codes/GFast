"""
Main script to run the GFast model on the GFP model.
"""
import random
import pandas as pd
import pickle
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
from math import floor
import itertools
import subprocess
from src.helper import Helper
from gfp_utils import read_fasta, select_aas, convert_encoding, aa_indices, sample_model, one_hot_encode, load_model, get_random_test_samples, qary_to_aa_encoding
from gfast.plot_utils import calculate_samples
from gfast.utils import load_data, save_data, get_qs, summarize_results, process_stdout, qary_vector_banned, dec_to_qary_vec, save_data4, qary_vec_to_dec, find_matrix_indices


model_weights = np.load('model/model_weights.npy')
sequence = read_fasta('model/avGFP.fasta')
sequence = sequence[:-1]
q = 20
t = 3
num_repeat = 39#int(args.num_repeat)
delays_method_source = "identity"
delays_method_channel = "nso"
hyperparam = True
# model = load_model('model/mlp.pt')
# Select the AAs that are above the threshold
threshold = 0.015
banned_indices = select_aas(sequence, threshold, model_weights)
n = 12
banned_indices_n = dict(itertools.islice(banned_indices.items(), n))
qs = get_qs(q, n, banned_indices_n)
print(qs)