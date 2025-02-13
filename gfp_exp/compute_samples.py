import numpy as np
import argparse
from src.helper import Helper
import os
import torch
from gfp_utils import load_model, one_hot_encode, sample_model, aa_indices, read_fasta
from gfast.utils import load_data, save_data, str_to_bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process q-ary indices arguments.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--num_subsample", type=int, required=True)
    parser.add_argument("--num_repeat", type=int, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--banned_indices_toggle", type=str, required=True)
    parser.add_argument("--banned_indices_path", type=str, required=False, default=None)

    args = parser.parse_args()

    n = args.n
    num_subsample = args.num_subsample
    num_repeat = args.num_repeat
    exp_dir = args.exp_dir
    batch_size = 512
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = load_model('model/mlp.pt', device=device)
    sequence = read_fasta('model/avGFP.fasta')
    sequence = sequence[:-1]
    banned_indices_toggle = str_to_bool(args.banned_indices_toggle)
    if args.banned_indices_path is not None:
        banned_indices = np.load(args.banned_indices_path, allow_pickle=True)
    
    for i in range(num_subsample):
        for j in range(num_repeat):
            query_indices_file = os.path.join(exp_dir, "train", "samples", "M{}_D{}_queryindices.pickle".format(i, j))
            query_indices = load_data(query_indices_file)
            sample_file = os.path.join(exp_dir, "train", "samples", "M{}_D{}.pickle".format(i, j))
            sample_file_mean = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
            folder_path = os.path.join(exp_dir, "train", "samples_mean")
            if not os.path.exists(os.path.join(folder_path)):
                os.makedirs(os.path.join(folder_path))

            if not os.path.isfile(sample_file):
                block_length = len(query_indices[0])
                samples = np.zeros((len(query_indices), block_length), dtype=complex)
                all_query_indices = np.concatenate(query_indices)
                if banned_indices_toggle == True:
                    seqs = aa_indices(n, sequence, all_query_indices, banned_indices=banned_indices, banned_indices_toggle=banned_indices_toggle)
                else:
                    seqs = aa_indices(n, sequence, all_query_indices, banned_indices_toggle=banned_indices_toggle)

                # One-hot encode sequences and sample
                one_hot_encoded_sequences = one_hot_encode(seqs)
                all_outputs = sample_model(model, one_hot_encoded_sequences, batch_size=batch_size, device=device)
                for k in range(len(query_indices)):
                    samples[k] = all_outputs[k * block_length: (k+1) * block_length]
                samples = samples.T
                save_data(samples, sample_file)
                save_data(samples, sample_file_mean)

    # Remove empirical mean
    mean_file = os.path.join(exp_dir, "train", "samples", "train_mean.npy")
    if not os.path.isfile(mean_file):
        all_samples = []
        for i in range(num_subsample):
            for j in range(num_repeat):
                sample_file = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
                samples = load_data(sample_file)
                samples = np.concatenate(samples)
                all_samples = np.concatenate([all_samples, samples])
        all_samples_mean = np.mean(all_samples)
        mean_file = os.path.join(exp_dir, "train", "samples", "train_mean.npy")
        np.save(mean_file, all_samples_mean)
    else:
        all_samples_mean = np.load(mean_file)
        for i in range(num_subsample):
            for j in range(num_repeat):
                sample_file_zeromean = os.path.join(exp_dir, "train", "samples", "M{}_D{}.pickle".format(i, j))
                sample_file = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
                samples = load_data(sample_file)
                samples_zeromean = samples - all_samples_mean
                save_data(samples_zeromean, sample_file_zeromean)

    # Sample test points from model
    query_indices_file = os.path.join(exp_dir, "test", "signal_t_queryindices.pickle")
    query_indices = load_data(query_indices_file)
    query_qaryindices_file = os.path.join(exp_dir, "test", "signal_t_query_qaryindices.pickle")
    query_qaryindices = load_data(query_qaryindices_file)
    sample_file = os.path.join(exp_dir, "test", "signal_t.pickle")
    sample_file_mean = os.path.join(exp_dir, "test", "signal_t_mean.pickle")
    if os.path.isfile(sample_file):
        if not os.path.isfile(sample_file_mean):
            samples = load_data(sample_file)
            save_data(samples, sample_file_mean)
    else:
        block_length = len(query_indices[0])
        samples = np.zeros((len(query_indices), block_length), dtype=complex)
        all_query_indices = query_indices.T
        if banned_indices_toggle == True:
            seqs = aa_indices(n, sequence, all_query_indices, banned_indices=banned_indices, banned_indices_toggle=banned_indices_toggle)
        else:
            seqs = aa_indices(n, sequence, all_query_indices, banned_indices_toggle=banned_indices_toggle)

        # One-hot encode sequences and sample
        one_hot_encoded_sequences = one_hot_encode(seqs)
        all_outputs = sample_model(model, one_hot_encoded_sequences, batch_size=batch_size, device=device)
        samples_dict = dict(zip(query_qaryindices, all_outputs))
        save_data(samples_dict, sample_file_mean)

    # Remove empirical mean
    sample_file_mean = os.path.join(exp_dir, "test", "signal_t_mean.pickle")
    sample_file = os.path.join(exp_dir, "test", "signal_t.pickle")
    samples_dict = load_data(sample_file_mean)

    all_values = list(samples_dict.values())
    all_values = np.array(all_values, dtype=complex) - all_samples_mean
    samples_dict = {key: value for key, value in zip(samples_dict.keys(), all_values)}
    save_data(samples_dict, sample_file)


def compute_scores(n, sequence, samples, banned_indices, banned_indices_toggle=True):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    model = load_model('model/mlp.pt', device=device)

    samples = samples.T
    if banned_indices_toggle == True:
        seqs = aa_indices(n, sequence, samples, banned_indices=banned_indices, banned_indices_toggle=banned_indices_toggle)
    else:
        seqs = aa_indices(n, sequence, all_query_indices, banned_indices_toggle=banned_indices_toggle)

    one_hot_encoded_sequences = one_hot_encode(seqs)
    all_outputs = sample_model(model, one_hot_encoded_sequences, batch_size=batch_size, device=device)
    return all_outputs