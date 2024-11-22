import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def load_data(data_path_x, data_path_y):
    dataX = np.load(data_path_x, allow_pickle=True)
    dataY = np.load(data_path_y, allow_pickle=True)

    keys = dataY[0].keys()
    dataY = dataY[dataX[:, 1, 0, 0] + 1e-2 < dataX[:, 1, -1, 0]]
    dataX = dataX[dataX[:, 1, 0, 0] + 1e-2 < dataX[:, 1, -1, 0]]
    dataY = np.array([[d[key] for key in keys] for d in dataY], dtype=np.float32)

    mean = np.mean(dataX, axis=0)
    std = np.std(dataX, axis=0)
    std[std == 0] = 1
    dataX = (dataX - mean) / std

    meany = np.mean(dataY, axis=0)
    stdy = np.std(dataY, axis=0)
    dataY = (dataY - meany) / stdy

    return dataX, dataY, mean, std, meany, stdy

def create_dataloaders(dataX, dataY, test_size=0.1, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=test_size, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
