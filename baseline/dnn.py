import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 定义网络层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(76 * 41 * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 14)

        # 512 is more worse
        # self.fc2 = nn.Linear(512, 512)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.fc3 = nn.Linear(512, 512)
        # self.bn3 = nn.BatchNorm1d(512)
        # self.fc4 = nn.Linear(512, 512)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.dropout = nn.Dropout(0.2)
        # self.output = nn.Linear(512, 14)

    def forward(self, x):
        # 前向传播
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

# 实例化模型
model = NeuralNetwork().to(device)

dataX = np.load("dmodel/data.npy",allow_pickle=True)
n = len(dataX)
dataY = np.load("dmodel/params.npy",allow_pickle=True)

keys = dataY[0].keys()
dataY = dataY[dataX[:, 1, 0, 0]+1e-2<dataX[:, 1, -1, 0]]
dataX = dataX[dataX[:, 1, 0, 0]+1e-2<dataX[:, 1, -1, 0]]
dataY = np.array([[d[key] for key in keys] for d in dataY], dtype=np.float32)

print(dataX.shape, dataY.shape)

# 归一化dataX,考虑除0的情况
mean = np.mean(dataX, axis=0)
std = np.std(dataX, axis=0)
std[std==0] = 1
dataX = (dataX - mean) / std
# 归一化dataY
meany = np.mean(dataY, axis=0)
stdy = np.std(dataY, axis=0)
dataY = (dataY - meany) / stdy

#定义将dataY还原的函数
def restore(data):
    return data * stdy + meany

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.1, random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 示例训练循环
losses = []
def train(model, criterion, optimizer, data_loader, num_epochs=300):
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到GPU
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model,"NN500.pth")

train(model, criterion, optimizer, train_loader)

# 测试集上的性能
def test(model, data_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到GPU
            outputs = model(inputs)
            #print(restore(outputs[0]),restore(targets[0]))
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

        mean_loss = total_loss / len(data_loader.dataset)
        print(f'Loss: {mean_loss:.4f}')

test(model, test_loader)

import json
with open("loss_nn.json","w") as f:
    json.dump(losses,f)
