import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from gfast.utils import qary_vector_banned, get_qs, get_signature
from math import floor     


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        
        # Input layer to bottleneck layer
        self.fc1 = nn.Linear(input_dim, 1)
        # Bottleneck layer to two neurons wide layer
        self.fc2 = nn.Linear(1, 2)
        self.sigmoid = nn.Sigmoid()
        # Two neurons wide layer to output layer
        self.fc3 = nn.Linear(2, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x


def read_fasta(file_path):
    """
    Reads a FASTA file and returns the sequence.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
    sequence = ''.join(sequence_lines)
    return sequence

import random
random.seed(42)
aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
def select_aas(sequence, threshold, model_weights):
    """
    Selects the AAs that are above the threshold based on the model weights.
    """
    aa_count = np.zeros(len(sequence), dtype=int)
    for position in range(len(model_weights)):
        num_significant_aas = np.sum(np.abs(model_weights[position,:]) > threshold)
        aa_count[position] = num_significant_aas

    banned_indices = {}
    for i, row in enumerate(model_weights):
        #only take relevant AAs, get rid of the ones with a very low weight
        indices_below = list(np.where(np.abs(row) < threshold)[0])
        if len(indices_below) >= 20 - 13:
            num = random.randint(3, 4)
            to_remove = random.sample(indices_below, num)
            indices_below = [x for x in indices_below if x not in to_remove]
        if len(indices_below) == len(row):
            #if ALL AAs are below the threshold we need at least one so we choose the wild type
            wild_type_aa = aa_dict[sequence[i]]
            indices_below.remove(wild_type_aa)
        if len(indices_below) != 0:
            banned_indices[i] = indices_below

    return banned_indices


seq_length = 237
def load_model(model_path, device='cpu'):
    model = MLP(seq_length*20).to(device)
    model.load_state_dict(torch.load(model_path))
    return model


amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
amino_to_index = {aa: i for i, aa in enumerate(amino_acids)}
one_hot_matrix = np.eye(len(amino_acids), dtype=int)


def one_hot_encode(seqs, pbar=False):
    num_sequences = len(seqs)
    encoded_length = seq_length * len(amino_acids)
    one_hot_encoded_sequences = np.zeros((num_sequences, encoded_length), dtype=int)

    if pbar:
        with tqdm(total=num_sequences, desc=f"One-hot encoding") as pbar:
            for idx, sequence in enumerate(seqs):
                indices = [amino_to_index[aa] for aa in sequence]  # Map amino acids to indices
                one_hot_encoded_sequences[idx, :len(sequence) * len(amino_acids)] = one_hot_matrix[indices].flatten()
                pbar.update(1)
    else:
        for idx, sequence in enumerate(seqs):
            indices = [amino_to_index[aa] for aa in sequence]  # Map amino acids to indices
            one_hot_encoded_sequences[idx, :len(sequence) * len(amino_acids)] = one_hot_matrix[indices].flatten()

    return one_hot_encoded_sequences


def sample_model(model, one_hot_encoded_sequences, batch_size=512, device='cpu'):
    one_hot_encoded_sequences = torch.tensor(one_hot_encoded_sequences, dtype=torch.float32).to(device)
    dataloader = DataLoader(one_hot_encoded_sequences, batch_size=batch_size, shuffle=False)
    all_outputs = []
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f"Computing samples") as pbar:
            for batch in dataloader:
                batch = batch.to(device)
                output = model(batch)
                all_outputs.append(output.cpu().numpy())
                pbar.update(1)
    all_outputs = np.concatenate(all_outputs, axis=None)
    return all_outputs


aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
num_encoding = {v: k for k, v in aa_dict.items()}
def aa_indices(n, sequence, all_query_indices, banned_indices_toggle=True, banned_indices=None):
    if banned_indices_toggle:
        # Adjust amino acid dictionary to only include the amino acids that are not in the banned indices
        site_aa_encoding = {}
        for key, value in banned_indices.items():
            encoding_reduced = {k: v for k, v in num_encoding.items() if k not in value}
            site_aa_encoding[key] = {new_key: v for new_key, (_, v) in enumerate(encoding_reduced.items())}
    if n < len(sequence):
        if banned_indices_toggle:
            indices_aas = [] 
            for row in all_query_indices:
                new_row = []
                for k, num in enumerate(row):
                    if k in banned_indices:
                        encoding_dict = site_aa_encoding[k]
                    else:
                        encoding_dict = num_encoding
                    encoded_num = encoding_dict[num]
                    new_row.append(encoded_num)
                indices_aas.append(new_row)
            indices_aas = [list(map(str, row)) for row in indices_aas]
        else:
            indices_aas = [[num_encoding[num] for num in row] for row in all_query_indices]
            indices_aas = [list(map(str, row)) for row in indices_aas]

        seq_padding = list(sequence)
        seqs = [''.join(row) + ''.join(seq_padding[len(row):]) for row in indices_aas]
        seqs = [list(seq) for seq in seqs]
    else:
        if banned_indices_toggle:
            indices_aas = []
            for row in all_query_indices:
                new_row = []
                for k, num in enumerate(row):
                    if k in banned_indices:
                        encoding_dict = site_aa_encoding[k]
                    else:
                        encoding_dict = num_encoding
                    encoded_num = encoding_dict[num]
                    new_row.append(encoded_num)
                indices_aas.append(new_row)
            indices_aas = [list(map(str, row)) for row in indices_aas]

        else:
            indices_aas = [[num_encoding[num] for num in row] for row in all_query_indices]
            indices_aas = [list(map(str, row)) for row in indices_aas]
        seqs = [''.join(row) for row in indices_aas]
        seqs = [list(seq) for seq in seqs]
    return seqs


def get_random_test_samples(q, n, banned_indices, num_samples=10000):
    total_qs = get_qs(q, n, banned_indices)
    N = np.prod(total_qs)
    base_inds_dec = [floor(random.uniform(0, 1) * N) for _ in range(num_samples)]
    query_indices = np.unique(np.array(base_inds_dec, dtype=object))
    random_samples = np.array(qary_vector_banned(query_indices, total_qs)).T
    return random_samples


def qary_to_aa_encoding(banned_indices):
    site_aa_encoding = {}
    for key, value in banned_indices.items():
        encoding_reduced = {k: v for k, v in num_encoding.items() if k not in value}
        site_aa_encoding[key] = {new_key: v for new_key, (_, v) in enumerate(encoding_reduced.items())}

    qary_to_aa_dict = {
        outer_key: {inner_key: aa_dict[inner_value] for inner_key, inner_value in inner_dict.items()}
        for outer_key, inner_dict in site_aa_encoding.items()
    }

    return qary_to_aa_dict


def convert_encoding(query_indices_banned, banned_indices):
    """
    Convert indices from banned space to regular space to get same AA sequences
    Takes input in same format as aa_indices (already transposed)
    """
    n_samples, seq_length = query_indices_banned.shape  # shape is transposed
    query_indices_regular = np.zeros_like(query_indices_banned)
    
    # Create banned encoding mapping for each position
    site_aa_encoding = {}
    for key, value in banned_indices.items():
        encoding_reduced = {k: v for k, v in num_encoding.items() if k not in value}
        site_aa_encoding[key] = {new_key: v for new_key, (_, v) in enumerate(encoding_reduced.items())}

    # Iterate through sequences maintaining transposed format
    for i in range(n_samples):
        for j in range(seq_length):
            banned_num = query_indices_banned[i,j]
            
            if j in banned_indices:
                # Get AA from banned encoding
                aa = site_aa_encoding[j][banned_num]
                # Look up regular number that gives same AA
                for reg_num, reg_aa in num_encoding.items():
                    if reg_aa == aa:
                        query_indices_regular[i,j] = reg_num
                        break
            else:
                query_indices_regular[i,j] = banned_num

    return query_indices_regular