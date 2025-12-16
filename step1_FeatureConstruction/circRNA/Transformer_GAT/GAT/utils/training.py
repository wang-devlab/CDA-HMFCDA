import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class AssociationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_feature_fusion(model, seq_features, edge_index, train_indices, val_indices, epochs=300, device='cpu'):
    """
    训练特征融合模型（支持图结构）
    
    参数:
    model (nn.Module): 要训练的模型
    seq_features (torch.Tensor): 序列特征张量
    edge_index (torch.Tensor): 图边索引
    train_indices (list): 训练集索引
    val_indices (list): 验证集索引
    epochs (int): 训练轮数
    device (str): 训练设备
    
    返回:
    tuple: (训练损失历史, 验证损失历史, 训练时间, 最佳模型状态)
    """
    model.to(device)
    seq_features = seq_features.to(device)
    edge_index = edge_index.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    train_loss_history = []
    val_loss_history = []
    start_time = time.time()
    
    print(f"Starting feature fusion training on {device} for {epochs} epochs...")
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        
        # 前向传播（使用整个图结构）
        _, reconstructed = model(seq_features, edge_index)
        
        # 计算重建损失（仅使用训练集节点）
        loss = F.mse_loss(reconstructed[train_indices], seq_features[train_indices])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        train_loss_history.append(train_loss)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            _, reconstructed = model(seq_features, edge_index)
            val_loss = F.mse_loss(reconstructed[val_indices], seq_features[val_indices]).item()
        
        val_loss_history.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
    
    training_time = time.time() - start_time
    print(f"Feature fusion training completed in {training_time:.2f} seconds")
    print(f"Best validation loss {best_val_loss:.6f} at epoch {best_epoch}")
    
    return train_loss_history, val_loss_history, training_time, best_model_state