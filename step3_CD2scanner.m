load("CD.mat");
M = CD;

t1 =1 ;
t2 =1 ;
for i = 1:size(M,1)
    for j = 1:size(M,2)
    	if M(i,j) >= 1
            ones_label(t1,:) = M(i,j);
            temp1 = [i j]; %%%%%i = row j = col
            ones_index(t1,:) = temp1;
            t1 = t1+1;
        else
            zeros_label(t2,:) = M(i,j);
            temp2 = [i j]; %%%%%i = row j = col
            zeros_index(t2,:) = temp2;
            t2 = t2+1;
        end
    end
end