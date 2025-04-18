import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import sys
sys.path.append("../..")
from tabular_exp.utils import convert_to_categorical, train_model
import json
import torch
torch.random.manual_seed(42)


df_datasets = pd.read_csv("datasets.csv")
datasets = df_datasets['Dataset'].values
ids = df_datasets['ID'].values
df_datasets['binary_variables'] = df_datasets['binary_variables'].apply(lambda x: eval(x) if isinstance(x, str) else None)
binary_variables = df_datasets['binary_variables'].values
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

for dataset, id, binary_variable in zip(datasets, ids, binary_variables):
    print(f"Preprocessing {dataset}")

    data = fetch_ucirepo(id=int(id)) 
    X = data.data.features 
    y = data.data.targets 

    if dataset == "breast_cancer":
        encoding_dict = {'no-recurrence-events': 0, 'recurrence-events': 1}
        y.loc[:, 'Class'] = y['Class'].map(encoding_dict)
    elif dataset == "car_evaluation":
        encoding_dict = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        y.loc[:, 'class'] = y['class'].map(encoding_dict)
        
    df, bin_mappings = convert_to_categorical(X, data.variables, y, binary=True, binary_variables=binary_variable)
    if dataset == "heart_disease":
        # Drop the last feature columm
        df = df.drop(df.columns[-2], axis=1)

    # Save files
    df.to_csv(f"{dataset}.csv", index=False)
    with open(f"{dataset}_bin_mappings.json", "w") as json_file:
        json.dump(bin_mappings, json_file, indent=4)

    # Train MLP
    train_model(df, dataset=dataset, device=device)
