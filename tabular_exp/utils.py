import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from tqdm import tqdm
import itertools
torch.random.manual_seed(42)


def convert_to_categorical(X: pd.DataFrame, variables: pd.DataFrame, y: pd.DataFrame, binary=False, binary_variables=None):
    """
    Converts numerical variables in X to categorical based on variables metadata.
    
    - Categorical variables remain unchanged.
    - Numerical variables with <5 unique values are converted to categorical.
    - Other numerical variables are discretized into 5 bins (or 4 if NaNs exist).

    Parameters:
        X (pd.DataFrame): Input features DataFrame.
        variables (pd.DataFrame): Metadata DataFrame defining categorical and numerical variables.
        y (pd.DataFrame): Target variable DataFrame.
        binary (bool): If True, convert target variable to binary.
        binary_variables (list): List of lists defining categories for binary conversion.

    Returns:
        pd.DataFrame: Transformed DataFrame with categorical encoding.
        dict: Bin mappings for each variable.
    """
    categorical_vars = set(variables.loc[variables["type"] == "Categorical", "name"])
    
    # Copy original DataFrame to modify
    X_transformed = X.copy()
    bin_mappings = {}

    for col in X.columns:
        if col in categorical_vars:
            continue  # Already categorical, leave it as is

        unique_vals = X[col].nunique()
        if unique_vals < 5:
            # Convert to categorical
            has_nans = X[col].isna().any()
            if has_nans:
                X_transformed[col] = X[col].astype('category').cat.codes + 2
                X_transformed[col] = np.where(X[col].isna(), 1, X_transformed[col])
                bin_mappings[col] = {1: "NaN"}
                for i, val in enumerate(X[col].dropna().unique()):
                    bin_mappings[col][i + 2] = val
            else:
                X_transformed[col] = X[col].astype('category').cat.codes + 1
                bin_mappings[col] = {i + 1: val for i, val in enumerate(X[col].unique())}
            continue

        # Determine number of bins
        min_val, max_val = X[col].min(skipna=True), X[col].max(skipna=True)
        has_nans = X[col].isna().any()
        num_bins = 4 if has_nans else 5  # 4 bins if NaNs exist, else 5 bins
        bins = np.linspace(min_val, max_val, num_bins + 1) 

        # Digitize values
        digitized_values = np.digitize(X[col], bins)
        digitized_values = np.where(digitized_values == len(bins), len(bins) - 1, digitized_values) # Force the max value to be the last bin

        if has_nans:
            # Shift bins up by 1 so we reserve 1 for NaNs
            digitized_values += 1
            digitized_values = np.where(X[col].isna(), 1, digitized_values)

        # Store bin ranges in the JSON dictionary
        bin_mappings[col] = {1: "NaN"} if has_nans else {}
        for i in range(len(bins) - 1):  # Iterate over bin edges
            category = i + (2 if has_nans else 1)  # Adjust category index if NaNs exist
            bin_mappings[col][category] = f"{bins[i]:.1f}-{bins[i + 1]:.1f}"

        X_transformed[col] = digitized_values

    # Convert target variable to binary if specified
    y = y.rename(columns={y.columns[0]: "y"})
    if binary:
        y_transformed = y.copy()
        if binary_variables:
            y_transformed['y'] = y['y'].apply(lambda val: 0 if val in binary_variables[0] else 1 if val in binary_variables[1] else val)
        else:
            y_transformed['y'] = np.where(y['y'] > 0, 1, y['y'])
    else:
        y_transformed = y.copy()

    # Merge transformed features and target variable
    result_df = pd.concat([X_transformed, y_transformed], axis=1)

    convert_int64(bin_mappings)

    return result_df, bin_mappings


def convert_int64(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_int64(value)
        elif isinstance(value, np.int64):
            d[key] = int(value)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def load_model(model_path, input_dim, device):
    model = MLP(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def train_model(df, epochs=500, batch_size=512, learning_rate=0.001, dataset='heart_disease', device='cpu'):
    """
    as the target variable)
        epochs (int): The number of epochs to train the model for.
        batch_size (int): The batch size to use for training.
        learning_rate (float): The learning rate for the optimizer.
        dataset (str): The name of the dataset.
        device (str): The device to use for training ('cpu' or 'cuda').
    """
    X = df.copy()
    y = df['y']

    # Check if the dataset has fewer than 100 samples for test set
    if len(df) * 0.2 < 100:
        test_size = min(100, len(df))
    else:
        test_size = int(len(df) * 0.2)

    # Ensure test_size is valid
    if test_size >= len(df):
        test_size = len(df) - 1

    # Train-test split
    y = y.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = X_train.drop(columns=['y'])
    X_test = X_test.drop(columns=['y'])
    X_train.to_csv(f'{dataset}_X_train.csv')
    X_test.to_csv(f'{dataset}_X_test.csv')

    # One-hot encode the categorical variables and save the encoder
    X = X.drop(columns=['y'])
    categories = []
    for col in X.columns:
        unique_values = X[col].unique()
        if type(unique_values[0]) == str:
            categories.append(unique_values)
        else:
            categories.append(sorted(unique_values))

    encoder = OneHotEncoder(categories=categories)
    encoder.fit(X)
    joblib.dump(encoder, f'../model/{dataset}_encoder.pkl')
    X_train_encoded = encoder.transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train_encoded.toarray(), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_encoded.toarray(), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    input_dim = X_train_tensor.shape[1]
    model = MLP(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    convergence_threshold = 1e-4

    # Training loop
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    previous_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        
        # Convergence criteria check based on training loss
        if abs(previous_loss - epoch_loss) < convergence_threshold:
            print("Convergence criteria met")
            break
        
        previous_loss = epoch_loss

    torch.save(model.state_dict(), f'../model/{dataset}_model.pth')

    print("Training complete and one-hot encoder saved.")

    # Testing the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted_labels = (test_outputs > 0.5).float()

    # Calculate accuracy in percentage
    accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test_tensor) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Find qs and save
    X_qary = X.to_numpy()
    qs = np.apply_along_axis(lambda x: len(set(x)), axis=0, arr=X_qary)
    np.save(f'../model/{dataset}_qs.npy', qs)



class ModelScorer:
    def __init__(self, model, qs, device='cpu'):
        self.model = model.to(device)
        self.qs = qs
        self.device = device

    def compute_average_score(self, row):
        indices_out_of_range = [i for i in range(len(row)) if row[i] >= self.qs[i]]
        possible_values = [list(range(self.qs[i])) for i in indices_out_of_range]
        possible_combinations = list(itertools.product(*possible_values))

        possible_rows = []
        for combination in possible_combinations:
            new_row = row.copy()
            for idx, val in zip(indices_out_of_range, combination):
                new_row[idx] = val
            possible_rows.append(new_row)

        possible_rows = self.one_hot_encode(np.array(possible_rows), self.qs)
        with torch.no_grad():
            batch_tensor = torch.tensor(possible_rows, dtype=torch.float32).to(self.device)
            outputs = self.model(batch_tensor)
            return outputs.mean(dim=0).cpu().numpy().item()
            
    def compute_scores(self, samples, batch_size=512):
        """
        Computes scores for each sample given the numpy array of samples and the trained model.

        Parameters:
            samples (np.ndarray): Numpy array of samples.
            batch_size (int): Number of samples per batch.

        Returns:
            np.ndarray: Numpy array of scores for each sample.
        """
        self.model.eval()
        scores = []

        valid_samples = []
        valid_indices = []
        average_scores = []

        for idx, row in tqdm(enumerate(samples), total=len(samples), desc="Computing scores for samples outside of alphabet space"):
            # If any element is outside of the alphabet at each index, compute the expectation of those features missing
            if any(row[j] >= self.qs[j] for j in range(len(row))):
                average_scores.append(self.compute_average_score(row))
            else:
                valid_samples.append(row)
                valid_indices.append(idx)

        # Compute scores for samples that are inside the alphabet space as normal
        if valid_samples:
            valid_samples = np.array(valid_samples)
            one_hot_samples = self.one_hot_encode(valid_samples, self.qs)

            with torch.no_grad():
                for i in tqdm(range(0, len(one_hot_samples), batch_size), total=int(np.ceil(len(one_hot_samples)/batch_size)), desc="Computing scores for samples within alphabet space"):
                    batch_samples = one_hot_samples[i:i + batch_size]
                    batch_tensor = torch.tensor(batch_samples, dtype=torch.float32).to(self.device)
                    outputs = self.model(batch_tensor)
                    batch_scores = outputs.cpu().numpy()
                    scores.append(batch_scores)

            # Combine the average scores with the scores for the samples we can compute directly
            scores = np.concatenate(scores, axis=0).reshape(-1)
            ordered_scores = np.zeros(len(samples))
            ordered_scores[valid_indices] = scores
            ordered_scores[np.setdiff1d(np.arange(len(samples)), valid_indices)] = average_scores
        else:
            ordered_scores = np.array(average_scores)

        return ordered_scores
        # valid_samples = np.array(valid_samples)
        # one_hot_samples = self.one_hot_encode(valid_samples, self.qs)

        # with torch.no_grad():
        #     for i in tqdm(range(0, len(one_hot_samples), batch_size), total=int(np.ceil(len(one_hot_samples)/batch_size)), desc="Computing scores for samples within alphabet space"):
        #         batch_samples = one_hot_samples[i:i + batch_size]
        #         batch_tensor = torch.tensor(batch_samples, dtype=torch.float32).to(self.device)
        #         outputs = self.model(batch_tensor)
        #         batch_scores = outputs.cpu().numpy()
        #         scores.append(batch_scores)

        # # Combine the average scores with the scores for the samples we can compute directly
        # scores = np.concatenate(scores, axis=0).reshape(-1)
        # ordered_scores = np.zeros(len(samples))
        # ordered_scores[valid_indices] = scores
        # ordered_scores[np.setdiff1d(np.arange(len(samples)), valid_indices)] = average_scores
        # return ordered_scores    

    def one_hot_encode(self, samples, qs):
        num_samples = len(samples)
        num_features = sum(qs)
        
        one_hot_encoded = np.zeros((num_samples, num_features), dtype=int)
        
        feature_index = 0
        for i, q in enumerate(qs):
            for j in range(num_samples):
                value = samples[j][i]
                one_hot_encoded[j][feature_index + value - 1] = 1
            feature_index += q
        
        return one_hot_encoded
    
    def get_samples_qary(self, samples):
        """
        Computes the total number of model queries needed to compute scores for all samples,
        accounting for expansion due to out-of-alphabet values.

        Parameters:
            samples (np.ndarray): Array of shape (num_samples, num_features)

        Returns:
            int: Total number of model queries required.
        """
        total_queries = 0

        for row in samples:
            indices_out_of_range = [i for i in range(len(row)) if row[i] >= self.qs[i]]
            if not indices_out_of_range:
                total_queries += 1
            else:
                # Compute the number of combinations required
                possible_values = [self.qs[i] for i in indices_out_of_range]
                num_combinations = np.prod(possible_values)
                total_queries += num_combinations

        return total_queries

    
