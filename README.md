# Efficient Algorithm for Sparse Fourier Transform of Generalized $q$-ary Functions

GFast is a an efficient algorithm to compute the sparse Fourier transform of generalized $q$-ary functions. This repository contains code from the article "Efficient Algorithm for Sparse Fourier Transform of Generalized $q$-ary Functions" by Darin Tsui*, Kunal Talreja*, and Amirali Aghazadeh. A link to the paper can be found [here](https://arxiv.org/abs/2501.12365).

\* Equal contributions

# Quick start

For an overview of GFast, as well as how to run the algorithm for your own functions, see ```tutorial.ipynb```.

## Description of folders

**gfast/:** GFast module for generalized $q$-ary functions.

**tabular_exp/:** Code needed to run NR-GFast for the multi-layer perceptron (MLP) trained on the heart disease dataset found [here](https://archive.ics.uci.edu/dataset/45/heart+disease). More information has been left in the folder.

**tabular_results/:** Folder containing the results of the heart disease experiments.

**gfp_exp/:** Code needed to run NR-GFast for the multi-layer perceptron (MLP) trained on the avGFP protein found in [this paper](https://www.nature.com/articles/nature17995). More information has been left in the folder.

**gfp_results/:** Folder containing the results of the GFP experiments.

**synt_exp/:** Code needed to run GFast and NR-GFast on synthetic data.

**synt_results/:** Folder containing the results of the synthetic experiments.

**tutorial_results/:** Folder containing the results of the tutorial.

# Figures

All figures in the paper and the code to generate them are found in ```figures.ipynb```.

