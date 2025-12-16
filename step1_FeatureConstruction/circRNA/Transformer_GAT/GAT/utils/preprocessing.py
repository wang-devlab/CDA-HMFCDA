import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_features(seq_features, assoc_matrix):
    """
    预处理特征数据
    
    参数:
    seq_features (np.ndarray): 序列特征
    assoc_matrix (np.ndarray): 关联矩阵
    
    返回:
    tuple: (处理后的序列特征, 处理后的关联矩阵, 标准化器)
    """
    # 确保数据是数值类型
    if seq_features.dtype != np.float32 and seq_features.dtype != np.float64:
        print(f"Converting sequence features from {seq_features.dtype} to float32")
        try:
            seq_features = seq_features.astype(np.float32)
        except Exception as e:
            print(f"转换序列特征失败: {e}")
            # 尝试处理混合类型数据
            print("尝试处理混合类型数据...")
            # 创建一个新的数组来存储转换后的值
            converted_seq = np.zeros_like(seq_features, dtype=np.float32)
            for i in range(seq_features.shape[0]):
                for j in range(seq_features.shape[1]):
                    try:
                        converted_seq[i, j] = float(seq_features[i, j])
                    except:
                        converted_seq[i, j] = 0.0
                        print(f"将位置 ({i}, {j}) 的值设置为0，原值为: {seq_features[i, j]}")
            seq_features = converted_seq
    
    # 确保关联矩阵是数值类型
    if assoc_matrix.dtype != np.float32 and assoc_matrix.dtype != np.float64:
        print(f"Converting association matrix from {assoc_matrix.dtype} to float32")
        try:
            assoc_matrix = assoc_matrix.astype(np.float32)
        except Exception as e:
            print(f"转换关联矩阵失败: {e}")
            # 尝试处理混合类型数据
            print("尝试处理混合类型数据...")
            converted_assoc = np.zeros_like(assoc_matrix, dtype=np.float32)
            for i in range(assoc_matrix.shape[0]):
                for j in range(assoc_matrix.shape[1]):
                    try:
                        converted_assoc[i, j] = float(assoc_matrix[i, j])
                    except:
                        converted_assoc[i, j] = 0.0
                        print(f"将位置 ({i}, {j}) 的值设置为0，原值为: {assoc_matrix[i, j]}")
            assoc_matrix = converted_assoc
    
    # 检查是否所有值都是数值型
    if np.any(np.isnan(seq_features)):
        print("警告: 序列特征包含NaN值，将填充为0")
        seq_features = np.nan_to_num(seq_features)
    
    if np.any(np.isnan(assoc_matrix)):
        print("警告: 关联矩阵包含NaN值，将填充为0")
        assoc_matrix = np.nan_to_num(assoc_matrix)
    
    # 标准化序列特征
    scaler = StandardScaler()
    try:
        seq_features_scaled = scaler.fit_transform(seq_features)
    except Exception as e:
        print(f"标准化序列特征时出错: {e}")
        # 如果还有问题，使用每列的均值和标准差手动标准化
        print("尝试手动标准化...")
        means = np.nanmean(seq_features, axis=0)
        stds = np.nanstd(seq_features, axis=0)
        # 避免除零
        stds[stds == 0] = 1.0
        seq_features_scaled = (seq_features - means) / stds
        seq_features_scaled = np.nan_to_num(seq_features_scaled)
    
    return seq_features_scaled, assoc_matrix, scaler