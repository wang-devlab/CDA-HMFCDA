import os
import time
import traceback
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# 导入自定义模块
from models.gat_fusion import GATFusionModel
from utils.data_loading import load_mat_data, save_enhanced_features
from utils.preprocessing import preprocess_features
from utils.graph_building import build_similarity_graph
from utils.training import train_feature_fusion
from utils.evaluation import evaluate_feature_quality


def main():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 配置参数 
    config = {
        # data输入，result输出配置
        "seq_file": os.path.join(current_dir, "data", "circRNA_NT_vectors.mat"),
        "seq_name": "circRNA_NT_vectors",
        "assoc_file": os.path.join(current_dir, "data", "CD.mat"),
        "assoc_name": "CD",
        "output_file": os.path.join(current_dir, "results", "NT_GATCD_features.mat"),
        "output_dir": os.path.join(current_dir, "results"),
        
        # 模型参数
        "gat_hidden_dim": 128,
        "gat_heads": 4,
        "fusion_dim": 256,
        
        # 训练参数
        "epochs": 500,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 32,
        
        # 图构建参数
        "similarity_threshold": 0.7,
        
        # 其他参数
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # 动态设置设备
        "random_seed": 42
    }
    
    # 设置随机种子
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    
    print(f"Using device: {config['device']}")
    
    # 检查文件是否存在
    if not os.path.exists(config["seq_file"]):
        print(f"警告: 序列特征文件 {config['seq_file']} 不存在")
    if not os.path.exists(config["assoc_file"]):
        print(f"警告: 关联矩阵文件 {config['assoc_file']} 不存在")
    
    try:
        # 1. 加载数据
        seq_features, assoc_matrix = load_mat_data(
            config["seq_file"], config["seq_name"], 
            config["assoc_file"], config["assoc_name"]
        )
        
        # 2. 预处理数据
        seq_features_scaled, assoc_matrix, scaler = preprocess_features(seq_features, assoc_matrix)
        
        # 转换为PyTorch张量
        seq_tensor = torch.tensor(seq_features_scaled, dtype=torch.float32)
        
        # 3. 构建图结构
        edge_index = build_similarity_graph(assoc_matrix, config["similarity_threshold"])
        
        # 4. 划分训练集和验证集索引
        n_samples = seq_features_scaled.shape[0]
        indices = np.arange(n_samples)
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=config["random_seed"]
        )
        
        print(f"Total samples: {n_samples}, Train: {len(train_indices)}, Validation: {len(val_indices)}")
        
        # 5. 初始化GAT特征融合模型
        model = GATFusionModel(
            seq_dim=seq_features_scaled.shape[1],
            gat_hidden_dim=config["gat_hidden_dim"],
            gat_heads=config["gat_heads"],
            fusion_dim=config["fusion_dim"]
        )
        
        # 6. 训练特征融合模型
        train_loss_history, val_loss_history, training_time, best_model_state = train_feature_fusion(
            model, 
            seq_tensor, 
            edge_index,
            train_indices,
            val_indices,
            epochs=config["epochs"],
            device=config["device"]
        )
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 7. 获取增强特征
        model.eval()
        with torch.no_grad():
            fused_features, _ = model(seq_tensor.to(config["device"]), edge_index.to(config["device"]))
            enhanced_features = fused_features.cpu().numpy()
        
        print(f"Enhanced features shape: {enhanced_features.shape}")
        
        # 8. 保存结果
        save_enhanced_features(enhanced_features, config["output_file"])
        
    #    
    except Exception as e:
        print(f"\nError occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()