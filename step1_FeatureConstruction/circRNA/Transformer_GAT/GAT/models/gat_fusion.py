import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATFusionModel(nn.Module):
    def __init__(self, seq_dim, gat_hidden_dim=128, gat_heads=4, fusion_dim=256):
        """
        基于图注意力网络的特征融合模型
        
        参数:
        seq_dim (int): 序列特征维度
        gat_hidden_dim (int): GAT隐藏层维度
        gat_heads (int): GAT注意力头数
        fusion_dim (int): 融合特征维度
        """
        super().__init__()
        
        # 序列特征编码器
        self.seq_encoder = nn.Sequential(
            nn.Linear(seq_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, gat_hidden_dim * gat_heads)
        )
        
        # 图注意力网络层
        self.gat1 = GATConv(
            gat_hidden_dim * gat_heads, 
            gat_hidden_dim, 
            heads=gat_heads,
            dropout=0.3
        )
        self.gat2 = GATConv(
            gat_hidden_dim * gat_heads, 
            fusion_dim, 
            heads=1,
            concat=False,
            dropout=0.3
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, fusion_dim)
        )
        
        # 重建层（确保信息不丢失）
        self.reconstruction = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, seq_dim)
        )
        
    def forward(self, seq_input, edge_index):
        # 编码序列特征
        seq_emb = self.seq_encoder(seq_input)
        
        # 图注意力网络处理
        x = F.elu(self.gat1(seq_emb, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        gat_out = self.gat2(x, edge_index)
        
        # 特征融合
        fused_features = self.fusion(gat_out)
        
        # 重建原始序列特征（自监督学习）
        reconstructed_seq = self.reconstruction(fused_features)
        
        return fused_features, reconstructed_seq