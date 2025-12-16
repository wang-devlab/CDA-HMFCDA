n_label = zeros_label;
n_index = zeros_index;
p_label = ones_label;
p_index = ones_index;


f_circl = C_fused; 
f_disea = D_fused;


n = length(p_label);
row_rank = randperm(size(n_index, 1));  
n_rank_index = n_index(row_rank,:); 

n_index = n_rank_index(1:length(p_label),:); 
n_label = n_label(1:length(p_label),:); 
n_label = n_label -1;
%%%ture fusion
for i = 1:size(p_index,1)
	temp = p_index(i,:);
	p_con_f(i,:) = [f_circl(temp(1),:) f_disea(temp(2),:)];
end
%%%false fusion
for i = 1:size(n_index,1)
	temp = n_index(i,:);
	n_con_f(i,:) = [f_circl(temp(1),:) f_disea(temp(2),:)];
end
p_sample = [p_label p_con_f];
n_sample = [n_label n_con_f];
name_p = [p_label p_index];
name_n = [n_label n_index];
