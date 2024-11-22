import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class AdapterLayer(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, adapter_dim=64):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.down_project = nn.Linear(d_model, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_project(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.up_project(x)
        return x + residual

class AdapterTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, adapter_dim=64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.adapter1 = AdapterLayer(d_model, dropout, adapter_dim)
        self.adapter2 = AdapterLayer(d_model, dropout, adapter_dim)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src2 = self.dropout1(src2)
        src = src + src2
        src = src + self.adapter1(src)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src2 = self.dropout2(src2)
        src = src + src2
        src = src + self.adapter2(src)
        
        return src

class DeviceParameterExtractor(nn.Module):
    def __init__(self, input_channels=2, sequence_length=41, num_curves=76, output_params=14, hidden_dim=256, num_heads=8, dropout_rate=0.2, adapter_dim=64):
        super(DeviceParameterExtractor, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_curves = num_curves
        
        self.curve_feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock(128),
        )
        
        self.temporal_attention = nn.MultiheadAttention(embed_dim=128, num_heads=num_heads, dropout=dropout_rate)
        
        encoder_layers = []
        for _ in range(4):
            encoder_layers.append(
                AdapterTransformerEncoderLayer(d_model=128, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_rate, adapter_dim=adapter_dim)
            )
        self.curve_transformer = nn.ModuleList(encoder_layers)
        
        self.feature_integration = nn.Sequential(
            nn.Linear(128 * num_curves, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.param_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.mean_head = nn.Linear(hidden_dim // 2, output_params)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, output_params),
            nn.Softplus()
        )
        
    def freeze_base_model(self):
        for param in self.parameters():
            param.requires_grad = False
            
        for module in self.modules():
            if isinstance(module, AdapterLayer):
                for param in module.parameters():
                    param.requires_grad = True
    
    def forward(self, x, return_uncertainty=True):
        batch_size = x.size(0)
        x = x.reshape(-1, self.input_channels, self.sequence_length)
        
        curve_features = self.curve_feature_extractor(x)
        
        curve_features = curve_features.permute(2, 0, 1)
        curve_features, _ = self.temporal_attention(curve_features, curve_features, curve_features)
        
        curve_features = curve_features.mean(dim=0)
        
        curve_features = curve_features.reshape(batch_size, self.num_curves, -1)
        
        curve_features = curve_features.permute(1, 0, 2)
        for encoder_layer in self.curve_transformer:
            curve_features = encoder_layer(curve_features)
        curve_features = curve_features.permute(1, 0, 2)
        
        flat_features = curve_features.reshape(batch_size, -1)
        integrated_features = self.feature_integration(flat_features)
        
        shared_features = self.param_predictor(integrated_features)
        param_mean = self.mean_head(shared_features)
        
        if return_uncertainty:
            param_var = self.var_head(shared_features)
            return {
                'mean': param_mean,
                'variance': param_var,
                'confidence_interval': 1.96 * torch.sqrt(param_var)
            }
        
        return param_mean
