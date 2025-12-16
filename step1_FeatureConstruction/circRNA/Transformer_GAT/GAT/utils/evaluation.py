import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split  

from models.association_predictor import AssociationPredictor
from utils.training import AssociationDataset

def train_association_predictor(features, assoc_matrix, feature_dim, n_diseases, device='cpu'):
    """
    训练关联预测模型评估特征质量
    
    参数:
    features (np.ndarray): 特征矩阵 (n_circ, feature_dim)
    assoc_matrix (np.ndarray): 关联矩阵 (n_circ, n_diseases)
    feature_dim (int): 特征维度
    n_diseases (int): 疾病数量
    device (str): 训练设备
    
    返回:
    dict: 评估指标 (AUC, AP, F1)
    """
    # 转换为PyTorch张量
    features_tensor = torch.tensor(features, dtype=torch.float32)
    assoc_tensor = torch.tensor(assoc_matrix, dtype=torch.float32)
    
    dataset = AssociationDataset(features_tensor, assoc_tensor)
    
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = AssociationPredictor(feature_dim, n_diseases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 训练模型
    best_auc = 0
    print("Training association predictor to evaluate feature quality...")
    
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 评估
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features_batch, labels_batch in test_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(features_batch)
                preds = torch.sigmoid(outputs)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels_batch.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # 计算指标
        auc_scores = []
        ap_scores = []
        f1_scores = []
        
        for i in range(n_diseases):
            if np.sum(all_labels[:, i]) > 0:  # 确保有正样本
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                ap = average_precision_score(all_labels[:, i], all_preds[:, i])
                f1 = f1_score(all_labels[:, i], (all_preds[:, i] > 0.5).astype(int), zero_division=0)
                
                auc_scores.append(auc)
                ap_scores.append(ap)
                f1_scores.append(f1)
        
        mean_auc = np.mean(auc_scores) if auc_scores else 0
        mean_ap = np.mean(ap_scores) if ap_scores else 0
        mean_f1 = np.mean(f1_scores) if f1_scores else 0
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_metrics = {
                'auc': mean_auc,
                'ap': mean_ap,
                'f1': mean_f1,
                'auc_scores': auc_scores,
                'ap_scores': ap_scores,
                'f1_scores': f1_scores
            }
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.4f} | AUC: {mean_auc:.4f} | "
                  f"AP: {mean_ap:.4f} | F1: {mean_f1:.4f}")
    
    print(f"Best AUC: {best_auc:.4f}")
    return best_metrics

def evaluate_feature_quality(features, assoc_matrix, feature_name, output_dir, device='cpu'):
    """评估特征质量并保存结果"""
    # 训练关联预测器
    metrics = train_association_predictor(
        features, assoc_matrix, 
        features.shape[1], assoc_matrix.shape[1],
        device=device
    )
    
    # 保存评估结果
    eval_file = os.path.join(output_dir, f'{feature_name}_evaluation.txt')
    with open(eval_file, 'w') as f:
        f.write(f"{feature_name} Feature Evaluation Results\n")
        f.write("===================================\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"Average Precision: {metrics['ap']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write("\nPer-disease AUC scores:\n")
        for i, auc in enumerate(metrics['auc_scores']):
            f.write(f"Disease {i+1}: {auc:.4f}\n")
    
    print(f"Evaluation results saved to {eval_file}")
    return metrics