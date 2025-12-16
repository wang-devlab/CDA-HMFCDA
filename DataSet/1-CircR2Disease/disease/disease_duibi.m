%% 特征矩阵质量评估工具箱
clear; clc; close all;

%% 1. 设置中文字体支持
try
    % 设置支持中文的字体
    list = listfonts; % 获取系统可用字体列表
    if any(contains(list, 'Microsoft YaHei'))
        fontName = 'Microsoft YaHei'; % Windows优先选择雅黑
    elseif any(contains(list, 'PingFang SC'))
        fontName = 'PingFang SC'; % macOS优先选择苹方
    elseif any(contains(list, 'SimHei'))
        fontName = 'SimHei'; % 备选黑体
    else
        fontName = 'Arial Unicode MS'; % 最后尝试Unicode字体
    end
    
    % 设置全局字体
    set(0, 'DefaultAxesFontName', fontName);
    set(0, 'DefaultTextFontName', fontName);
    set(0, 'DefaultUicontrolFontName', fontName);
    set(0, 'DefaultUitableFontName', fontName);
    set(0, 'DefaultUipanelFontName', fontName);
catch
    warning('字体设置失败，中文显示可能不正常');
end

%% 1. 数据加载
try
    % 加载疾病特征矩阵
    disease_data = load('disease_vectors.mat');
    
    % 检查字段名并提取矩阵
    if isfield(disease_data, 'disease_vectors')
        disease_matrix = disease_data.disease_vectors;
    elseif isfield(disease_data, 'disease_vector')
        disease_matrix = disease_data.disease_vector;
    else
        error('disease_vectors.mat 中未找到特征矩阵');
    end
    
    % 加载MeSH特征矩阵
    mesh_data = load('MeSHSemanticSimilarity.mat');
    
    % 检查字段名并提取矩阵
    if isfield(mesh_data, 'MeSHSemanticSimilarity')
        mesh_matrix = mesh_data.MeSHSemanticSimilarity;
    elseif isfield(mesh_data, 'mesh_matrix')
        mesh_matrix = mesh_data.mesh_matrix;
    else
        error('MeSHSemanticSimilarity.mat 中未找到特征矩阵');
    end
catch ME
    error('文件加载失败: %s\n请检查文件名和路径', ME.message);
end

% 验证矩阵尺寸
if ~isequal(size(disease_matrix), [100, 100])
    disease_matrix = disease_matrix(1:100, 1:100);
    warning('disease matrix 尺寸已调整为 100×100');
end

if ~isequal(size(mesh_matrix), [100, 100])
    mesh_matrix = mesh_matrix(1:100, 1:100);
    warning('MeSH matrix 尺寸已调整为 100×100');
end



%% 3. 评估两个矩阵
disease_report = evaluate_matrix(disease_matrix, 'disease_vectors');
mesh_report = evaluate_matrix(mesh_matrix, 'MeSHSemanticSimilarity');

%% 4. 结果可视化
% 创建对比表格
metrics = {
    'ConditionNumber', '条件数', '数值稳定性(值越小越好)', '%.2e';
    'CovConditionNumber', '协方差条件数', '特征独立性(值越小越好)', '%.2e';
    'MeanCorrelation', '平均特征相关性', '特征冗余度(<0.3优)', '%.4f';
    'NumComponents95', '95%方差所需主成分数', '信息丰富度(>50优)', '%d';
    'MeanKurtosis', '平均峰度', '分布特性(接近3优)', '%.2f';
    'MeanEntropy', '平均信息熵', '信息量(值越大越好)', '%.2f';
    'Sparsity', '稀疏度', '稀疏性(80%优)', '%.4f'
};

% 创建结果表
results = cell(size(metrics, 1), 4);
for i = 1:size(metrics, 1)
    field_name = metrics{i, 1};
    metric_name = metrics{i, 2};
    description = metrics{i, 3};
    format_str = metrics{i, 4};
    
    disease_val = disease_report.(field_name);
    mesh_val = mesh_report.(field_name);
    
    results{i, 1} = metric_name;
    results{i, 2} = description;
    
    if isnan(disease_val)
        results{i, 3} = 'N/A';
    else
        results{i, 3} = sprintf(format_str, disease_val);
    end
    
    if isnan(mesh_val)
        results{i, 4} = 'N/A';
    else
        results{i, 4} = sprintf(format_str, mesh_val);
    end
end

% 显示结果表格
fprintf('\n=================== 特征矩阵质量评估报告 ===================\n');
fprintf('%-25s %-35s %-20s %-20s\n', '指标', '描述', 'disease_vectors', 'MeSH_matrix');
fprintf('==============================================================================\n');
for i = 1:size(results, 1)
    fprintf('%-25s %-35s %-20s %-20s\n', results{i, 1}, results{i, 2}, results{i, 3}, results{i, 4});
end

% PCA可视化
% 修改后代码：
if all(~isnan(disease_report.PCAExplained)) && all(~isnan(mesh_report.PCAExplained))
    figure('Position', [100, 100, 1200, 500], 'Name', 'PCA方差解释率');
    
    % 获取实际主成分数量
    n_comp_disease = length(disease_report.PCAExplained);
    n_comp_mesh = length(mesh_report.PCAExplained);
    
    subplot(1, 2, 1);
    plot(1:n_comp_disease, disease_report.PCAExplained, 'b-o', 'LineWidth', 1.5); % 使用实际长度
    hold on;
    plot([disease_report.NumComponents95, disease_report.NumComponents95], [0, 1], 'r--');
    title('disease matrix PCA方差解释率');
    xlabel('主成分数量');
    ylabel('累积解释方差');
    legend('累积方差', '95%阈值', 'Location', 'southeast');
    grid on;
    ylim([0, 1.05]);
    xlim([1, max(n_comp_disease, n_comp_mesh)]); % 动态设置坐标范围

    subplot(1, 2, 2);
    plot(1:n_comp_mesh, mesh_report.PCAExplained, 'r-o', 'LineWidth', 1.5); % 使用实际长度
    hold on;
    plot([mesh_report.NumComponents95, mesh_report.NumComponents95], [0, 1], 'b--');
    title('MeSH matrix PCA方差解释率');
    xlabel('主成分数量');
    ylabel('累积解释方差');
    legend('累积方差', '95%阈值', 'Location', 'southeast');
    grid on;
    ylim([0, 1.05]);
    xlim([1, max(n_comp_disease, n_comp_mesh)]); % 动态设置坐标范围
else
    warning('无法显示PCA图表：一个或两个矩阵的PCA结果无效');
end

% 特征相关性分布可视化
try
    figure('Position', [100, 100, 1000, 400], 'Name', '特征相关性分布');
    subplot(1, 2, 1);
    corr_disease = corrcoef(disease_matrix);
    histogram(corr_disease(:), 50, 'FaceColor', 'b');
    title('disease matrix特征相关性分布');
    xlabel('相关系数');
    ylabel('频数');
    xlim([-1, 1]);
    
    subplot(1, 2, 2);
    corr_mesh = corrcoef(mesh_matrix);
    histogram(corr_mesh(:), 50, 'FaceColor', 'r');
    title('MeSH matrix特征相关性分布');
    xlabel('相关系数');
    ylabel('频数');
    xlim([-1, 1]);
catch
    warning('无法显示相关性分布图表');
end

%% 6. 计算综合质量评分
disease_score = calculate_quality_score(disease_report);
mesh_score = calculate_quality_score(mesh_report);

fprintf('\n=================== 综合质量评分 ===================\n');
fprintf('disease_vectors 质量评分: %.2f/1.00\n', disease_score);
fprintf('MeSH_matrix 质量评分: %.2f/1.00\n', mesh_score);

% 显示质量建议
for i = 1:size(disease_score,2)
    if disease_score(i) > mesh_score(i)
        fprintf('\n推荐结论%d: disease_vectors 是更优的特征矩阵 (%.2f > %.2f)\n',i, disease_score(i), mesh_score(i));
    else
        fprintf('\n推荐结论%d: MeSH_matrix 是更优的特征矩阵 (%.2f > %.2f)\n',i, mesh_score(i), disease_score(i));
    end
end

%% 2. 定义评估函数
function report = evaluate_matrix(matrix, matrix_name)
    % 初始化报告结构
    report = struct();
    report.Name = matrix_name;
    report.Size = size(matrix);
    
    % 1. 数值稳定性
    report.ConditionNumber = cond(matrix);
    report.FrobeniusNorm = norm(matrix, 'fro');
    
    % 2. 特征独立性
    try
        corr_matrix = corrcoef(matrix);
        % 取上三角（不含对角线）
        mask = triu(true(size(corr_matrix)), 1);
        corr_values = corr_matrix(mask);
        report.MeanCorrelation = mean(abs(corr_values));
    catch
        report.MeanCorrelation = NaN;
        warning('无法计算 %s 的相关性矩阵', matrix_name);
    end
    
    try
        cov_matrix = cov(matrix);
        report.CovConditionNumber = cond(cov_matrix);
    catch
        report.CovConditionNumber = NaN;
        warning('无法计算 %s 的协方差矩阵条件数', matrix_name);
    end
    
    % 3. 信息量分析 (PCA)
    try
        [~, ~, latent] = pca(matrix);
        explained = cumsum(latent)/sum(latent);
        report.NumComponents95 = find(explained >= 0.95, 1);
        report.PCAExplained = explained;
    catch
        report.NumComponents95 = NaN;
        report.PCAExplained = NaN(100, 1);
        warning('无法对 %s 进行PCA分析', matrix_name);
    end
    
    % 4. 分布特性
    try
        kurtosis_vals = kurtosis(matrix);
        report.MeanKurtosis = mean(kurtosis_vals);
    catch
        report.MeanKurtosis = NaN;
    end
    
    % 计算信息熵（分箱法）
    entropy_vals = zeros(1, size(matrix, 2));
    valid_columns = 0;
    for i = 1:size(matrix, 2)
        col = matrix(:, i);
        if all(~isnan(col)) && ~all(col == col(1)) % 检查是否非常数
            [counts, ~] = histcounts(col, 100, 'Normalization', 'probability');
            counts = counts(counts > 0); % 移除零概率
            entropy_vals(i) = -sum(counts.*log2(counts));
            valid_columns = valid_columns + 1;
        else
            entropy_vals(i) = 0;
        end
    end
    
    if valid_columns > 0
        report.MeanEntropy = mean(entropy_vals(entropy_vals > 0));
    else
        report.MeanEntropy = 0;
    end
    
    % 5. 稀疏性：计算零元素占比（真正的稀疏度）
    zero_count = numel(matrix) - nnz(matrix);
    report.Sparsity = zero_count / numel(matrix);  % 零元素占比
end

%% 5. 质量评分函数
function score = calculate_quality_score(report)
    weights = [0.25,  0.20,  0.15,  0.15, 0.10, 0.10,  0.05]; % 7个指标的权重总和为1
    
    % 获取指标值
    values = [
        report.ConditionNumber,
        report.CovConditionNumber,
        report.MeanCorrelation,
        report.NumComponents95,
        report.MeanKurtosis,
        report.MeanEntropy,
        report.Sparsity
    ];
    
    % 检查是否有NaN值
    if any(isnan(values([1, 2, 3, 5, 6, 7]))) || isnan(values(4))
        score = NaN;
        return;
    end
    
    % 标准化各项指标 (0-1范围，1表示最优)
    metrics = [
        1/(1+log10(max(1, values(1)))),...         % 对数压缩
        1/(1+log10(max(1, values(2)))),...        % 对数压缩
        1 - min(1, values(3)/0.5),...              % 相关性<0.5得高分
        min(1, values(4)/70),...                   % 70个成分得满分
        1 - min(1, abs(values(5) - 3)/2),...       % 峰度接近3得高分
        min(1, values(6)/8), ...                   % 熵=8得满分
        1 - abs(values(7) - 0.60)                % 20%稀疏度最佳
    ];
    
    % 加权得分
    score = sum(weights .* metrics);
end