import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=2, hidden_size=128, num_layers=2, output_dim=14, dropout=0.2):
        super().__init__()
        self.input_channels = input_channels
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # 计算CNN输出后的特征维度
        self.cnn_feature_size = 5248
        
        # 全连接层将CNN特征转换为LSTM输入
        self.fc_cnn = nn.Sequential(
            nn.Linear(self.cnn_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2是因为双向LSTM
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = 41
        
        x = x.reshape(-1, self.input_channels, seq_length)
        
        # CNN特征提取
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)
        
        # 转换CNN特征
        features = self.fc_cnn(cnn_out)
        
        # 重塑为LSTM输入格式 (batch, seq_len, features)
        features = features.reshape(batch_size, 76, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(features)
        
        # 我们只使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 通过输出层得到最终预测
        output = self.fc_out(last_output)
        
        return output

# Model instantiation
model = CNNLSTM(
        input_channels=2,      # 输入通道数
        hidden_size=128,       # LSTM隐藏层大小
        num_layers=2,          # LSTM层数
        output_dim=14,         # 输出维度
        dropout=0.2            # dropout率
    ).to(device)

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

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    torch.save(model,"cnnlstm300.pth")

#train(model, criterion, optimizer, train_loader)

def train_model_val(model, data_loader, test_loader, num_epochs=300, patience=50):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion2 = nn.MSELoss()
    losses = []  # 用于存储每个epoch的损失值
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for curves, params in data_loader:
            curves, params = curves.to(device), params.to(device)  # 将数据转移到GPU
            
            optimizer.zero_grad()
            
            # 前向传播
            pred_dict = model(curves)
            
            # 计算损失
            loss = criterion(pred_dict, params)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            #epoch_loss += loss_dict["mse_loss"]
        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        model.eval()
        loss_val = []
        val_losses = []
        with torch.no_grad():
            for batch_curves, batch_params in test_loader:
                batch_curves,batch_params = batch_curves.to(device),batch_params.to(device)
                pred_dict = model(batch_curves)
                val_loss = criterion2(pred_dict, batch_params)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        loss_val.append(avg_val_loss)
        print(f"test_loss: {avg_val_loss}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_modelcnn.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return losses,loss_val  # 返回损失值列表

losses,losses_val = train_model_val(model, train_loader, test_loader)

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


