% data = [p_sample; n_sample]; 
% index = [name_p; name_n]; 
% KFold=5;% 
% [test_index,test_data,train_data,test_lab,train_lab] = KFoldCrossValidation2(index,data,KFold);% K’€Ωª≤Ê—È÷§
% %%%% MLP model %%%%
% [Prediction,Accuracy,Dec_values] = MLP(test_data,test_lab,train_data,train_lab,KFold);

%%%% result%%%%
[VACC,VSN,VPE,VMCC,VF1] = roc(Prediction,test_lab',KFold);
[VAUC]=plotrocZOOM(test_lab,Dec_values,KFold);%roc

%
ValACC=mean(cell2mat(VACC));
ValPE=mean(cell2mat(VPE));
ValSN=mean(cell2mat(VSN));
ValMCC=mean(cell2mat(VMCC));
ValF1=mean(cell2mat(VF1));
ValAUC=mean(VAUC);
%
stdACC=std(cell2mat(VACC));
stdPE=std(cell2mat(VPE));
stdSN=std(cell2mat(VSN));
stdMCC=std(cell2mat(VMCC));
stdF1=std(cell2mat(VF1));
stdAUC=std(VAUC);

