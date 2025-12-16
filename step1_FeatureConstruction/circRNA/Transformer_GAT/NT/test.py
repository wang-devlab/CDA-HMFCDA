from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import scipy.io
import numpy as np

# 指定本地模型路径
model_path = "./nt-model"  # 替换为您本地模型的实际路径

# 导入模型（从本地路径）
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForMaskedLM.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path, ignore_mismatched_sizes=True)

# 选择输入序列填充的长度
max_length = tokenizer.model_max_length

# 加载.mat文件序列并对其进行分词
mat_data = scipy.io.loadmat('CircR2Disease_circRNA_list_seq.mat')  # 替换为您的文件名
circRNA_data = mat_data['circRNA_list_seq']  # 取到'circRNA_list_seq'中的数据
circRNA_names = circRNA_data[:, 0]  # 第一列名为'circRNA_names'
circRNA_sequences = circRNA_data[:, 1]  # 第二列名为'circRNA_sequences'

# 确保所有序列都是字符串类型
circRNA_sequences = [str(seq[0]) if isinstance(seq, np.ndarray) else str(seq) for seq in circRNA_sequences]
circRNA_names = [str(name[0]) if isinstance(name, np.ndarray) else str(name) for name in circRNA_names]

print(f"共加载 {len(circRNA_sequences)} 条circRNA序列")

# 分批处理序列以避免内存不足
batch_size = 8  # 根据您的GPU内存调整批次大小
all_embeddings = []

for i in range(0, len(circRNA_sequences), batch_size):
    batch_sequences = circRNA_sequences[i:i+batch_size]
    
    # 对批次中的序列进行分词
    tokens_ids = tokenizer.batch_encode_plus(
        batch_sequences, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=max_length,
        truncation=True  # 确保超长序列被截断
    )["input_ids"]
    
    # 计算注意力掩码
    attention_mask = tokens_ids != tokenizer.pad_token_id
    
    # 计算嵌入
    with torch.no_grad():  # 禁用梯度计算以节省内存
        torch_outs = model(
            tokens_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    
    # 获取序列嵌入
    embeddings = torch_outs['hidden_states'][-1].detach()
    
    # 添加嵌入维度轴
    attention_mask_expanded = torch.unsqueeze(attention_mask, dim=-1)
    
    # 计算每个序列的平均嵌入
    mean_sequence_embeddings = torch.sum(attention_mask_expanded * embeddings, axis=1) / torch.sum(attention_mask_expanded, axis=1)
    
    # 收集所有批次的嵌入
    all_embeddings.append(mean_sequence_embeddings.numpy())
    
    print(f"已处理 {min(i+batch_size, len(circRNA_sequences))}/{len(circRNA_sequences)} 条序列")

# 合并所有批次的嵌入
all_embeddings = np.vstack(all_embeddings)
print(f"最终嵌入矩阵形状: {all_embeddings.shape}")  # 应该是 (序列数量, 隐藏层大小)

# 将名称和向量保存为MATLAB兼容的格式
# 确保名称是MATLAB兼容的格式（字符串数组）
rna_names_flat = np.array(circRNA_names, dtype=object)

# 创建结构化数组确保MATLAB兼容性
output_data = {
    'circRNA_names': rna_names_flat,  # 按行保存的名称
    'circRNA_NT_vectors': all_embeddings  # 向量矩阵
}

# 保存为.mat文件
scipy.io.savemat('circRNA_NT_vectors.mat', output_data)
print("数据已保存到 circRNA_NT_vectors.mat")

