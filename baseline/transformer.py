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



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_shape, output_dim, embed_size=256, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        # 计算输入维度
        self.seq_len = input_shape[1]  # 76
        self.features = input_shape[2] * input_shape[3]  # 41 * 2
        
        # 输入映射层
        self.input_projection = nn.Linear(self.features, embed_size)
        
        # Position Encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, embed_size))
        
        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embed_size * self.seq_len, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 重塑输入: (batch, seq_len, features)
        x = x.reshape(batch_size, self.seq_len, -1)
        
        # 投影到embedding维度
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # 展平并通过输出层
        x = x.reshape(batch_size, -1)
        x = self.output_layer(x)
        
        return x

# Example hyperparameters (adjust based on your needs)
input_dim = 76 * 41 * 2   # Input feature dimension
embed_size = 512  # Embedding size of the transformer
num_heads = 8     # Number of attention heads
ff_hidden_size = 256 # Hidden size of the FFN layer
num_layers = 4    # Number of transformer blocks
output_dim = 14   # Number of output classes or features (adjust based on task)

# Model instantiation
# model = Transformer(input_dim, embed_size, num_heads, ff_hidden_size, num_layers, output_dim).to(device)
input_shape = (None, 76, 41, 2)
output_dim = 14
model = Transformer(input_shape, output_dim).to(device)
# Example input tensor with shape [batch_size, seq_length, input_dim]
# x = torch.randn(32, 197, input_dim)

# # Forward pass
# output = model(x).to(device)
# print(output.shape)  # Should be [batch_size, seq_length, output_dim]

# 实例化模型
# model = TransformerX().to(device)

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
#criterion = nn.L1Loss()
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
    torch.save(model,"Transformer300.pth")

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
