#!/bin/bash

python run_test_samples_vs_nmse.py \
--q 7 --n 20 --b 6 --delta 40 50 60 --sparsity 1000 --snr 100 --a 1 --iters 3 --removal uniform --num_permutations 5

python run_test_samples_vs_nmse_nso.py \
--q 7 --n 20 --b 5 --delta 40 50 60 --sparsity 1000 --snr 10 --a 1 --iters 3 --removal uniform --num_permutations 2