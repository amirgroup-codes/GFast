import numpy as np
import argparse
from src.helper import Helper
import os
from pathlib import Path
from gfast.utils import str_to_bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process q-ary indices arguments.")
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--delta", type=int, required=True)
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--num_subsample", type=int, required=True)
    parser.add_argument("--num_repeat", type=int, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--banned_indices_path", type=str, required=False)
    parser.add_argument("--banned_indices_toggle", type=str, required=True)
    parser.add_argument("--delays_method_channel", type=str, required=False)

    args = parser.parse_args()
    if args.delays_method_channel:
        delays_method_channel = args.delays_method_channel
    else:
        delays_method_channel = "identity"
    if str_to_bool(args.banned_indices_toggle):
        banned_indices_n = np.load(args.banned_indices_path, allow_pickle=True)
    else:
        banned_indices_n = {}

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    query_args = {
        "query_method": 'generate_samples',
        "train_samples": 'simple',
        "method": "generate_samples",
        "num_subsample": args.num_subsample,
        "num_repeat": args.num_repeat,
        "b": args.b,
        "folder": exp_dir,
        "delays_method_channel": delays_method_channel
    }
    signal_args = {
        "n":args.n,
        "q":args.q,
        "query_args":query_args,
        "len_seq":args.n,
        "banned_indices_toggle": str_to_bool(args.banned_indices_toggle),
        "banned_indices": banned_indices_n,
        "delays_method_channel": delays_method_channel
        }
    test_args = {
            "n_samples": 10000,
            "method": "generate_samples"
        }
    file_path = os.path.join(exp_dir, 'train', 'samples')
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(exp_dir, 'test')
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    helper = Helper(signal_args=signal_args, methods=["gfast"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
    