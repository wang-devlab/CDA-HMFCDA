import numpy as np
import torch

def build_similarity_graph(assoc_matrix, threshold=0.7):
    """
    从关联矩阵构建circRNA-circRNA相似图
    
    参数:
    assoc_matrix (np.ndarray): 关联矩阵 (n_circ, n_diseases)
    threshold (float): 相似度阈值
    
    返回:
    torch.Tensor: 边索引 (2, num_edges)
    """
    # 计算circRNA之间的余弦相似度
    norm = np.linalg.norm(assoc_matrix, axis=1, keepdims=True)
    normalized = assoc_matrix / np.where(norm > 0, norm, 1)
    sim_matrix = normalized @ normalized.T
    
    # 应用阈值创建邻接矩阵
    adj_matrix = (sim_matrix > threshold).astype(np.float32)
    
    # 确保没有自环
    np.fill_diagonal(adj_matrix, 0)
    
    # 转换为边索引格式
    edge_index = torch.tensor(np.array(np.where(adj_matrix)), dtype=torch.long)
    
    print(f"构建相似图: 节点数={assoc_matrix.shape[0]}, 边数={edge_index.shape[1]}")
    return edge_index