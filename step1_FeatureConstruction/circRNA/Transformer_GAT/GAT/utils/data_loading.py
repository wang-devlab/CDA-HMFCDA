import os
import scipy.io

def load_mat_data(seq_file, seq_name, assoc_file, assoc_name):
    """
    从MATLAB文件中加载数据
    
    参数:
    seq_file (str): 序列特征文件路径
    seq_name (str): 序列特征变量名
    assoc_file (str): 关联矩阵文件路径
    assoc_name (str): 关联矩阵变量名
    
    返回:
    tuple: (seq_features, assoc_matrix)
    """
    # 检查文件是否存在
    if not os.path.exists(seq_file):
        raise FileNotFoundError(f"序列特征文件 {seq_file} 不存在")
    if not os.path.exists(assoc_file):
        raise FileNotFoundError(f"关联矩阵文件 {assoc_file} 不存在")
    
    print(f"Loading sequence features from {seq_file} with variable name '{seq_name}'...")
    seq_data = scipy.io.loadmat(seq_file)
    
    # 检查变量是否存在
    if seq_name not in seq_data:
        available_keys = [k for k in seq_data.keys() if not k.startswith('__')]
        raise KeyError(f"变量 '{seq_name}' 不存在于文件中。可用变量: {available_keys}")
    
    seq_features = seq_data[seq_name]
    print(f"Sequence features shape: {seq_features.shape}")
    
    print(f"Loading association matrix from {assoc_file} with variable name '{assoc_name}'...")
    assoc_data = scipy.io.loadmat(assoc_file)
    
    # 检查变量是否存在
    if assoc_name not in assoc_data:
        available_keys = [k for k in assoc_data.keys() if not k.startswith('__')]
        raise KeyError(f"变量 '{assoc_name}' 不存在于文件中。可用变量: {available_keys}")
    
    assoc_matrix = assoc_data[assoc_name]
    print(f"Association matrix shape: {assoc_matrix.shape}")
    
    return seq_features, assoc_matrix

def save_enhanced_features(enhanced_features, output_file):
    """
    保存增强特征到MATLAB文件
    
    参数:
    enhanced_features (np.ndarray): 增强的特征矩阵
    output_file (str): 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir != '' and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # 从文件名中提取变量名（去掉扩展名）
    variable_name = os.path.basename(output_file).split('.')[0]
    # 保存为MATLAB格式，使用与文件名相同的变量名
    scipy.io.savemat(output_file, {variable_name: enhanced_features})
    print(f"Enhanced features saved to {output_file}")