import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path, train_split=0.8):
    selected_feature = [
        'G1', 'G2', 'failures', 'age', 'Fedu', 'Medu',
        'goout', 'traveltime', 'Walc'
    ]
    target = 'Pass'

    # Load the dataset
    data = pd.read_csv(file_path)
    # Convert G3 to binary
    data[target] = (data['G3'] >= 10).astype(int)
    # Select features and target
    df = data[selected_feature + [target]]

    # transform 
    df['failures'] = np.log1p(df['failures'])
    df['traveltime'] = np.log1p(df['traveltime'])

    # Normalize 
    scaler = StandardScaler()
    df[selected_feature] = scaler.fit_transform(df[selected_feature])


    # Split the data into features and target
    X = df[selected_feature].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32) 
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    return train_dataset, test_dataset


if __name__ == "__main__":
    file_path = '../../data/Math-Students.csv'
    train_dataset, test_dataset = load_data(file_path)
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    print("First sample in train dataset:", train_dataset[0])
    print("First sample in test dataset:", test_dataset[0])


    '''
    Train dataset size: 319
    Test dataset size: 80
    First sample in train dataset: (tensor([ 1.2444,  0.8848, -0.4840, -1.3389, -0.4914,  1.1307, -0.9874, -0.6780,-1.0063]), tensor(1.))
    First sample in test dataset: (tensor([2.1505, 1.9497, 1.2887, 0.2277, 1.3412, 1.1307, 0.8115, 0.9676, 0.5560]), tensor(1.))
    '''