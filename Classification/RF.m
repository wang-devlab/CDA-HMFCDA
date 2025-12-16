 function [Prediction,Accuracy,Dec_values]=RF(test_data,test_lab,train_data,train_lab,KFold)

    Dec_values{1,KFold}=[];
    Prediction{1,KFold}=[];
    Accuracy{1,KFold}=[];
for i=1:KFold
    testdata=test_data{i,1};
    test1ab=test_lab{i,1};
    traindata=train_data{i,1};
    train1ab=train_lab{i,1};
    model{i,1}=TreeBagger(100,traindata,train1ab,'Method','classification');
    
    [prediction,dec_values]=predict(model{i,1},testdata);
    rf_temp=char(prediction);       %% here, RF's prediction needs cell2mat
    rf_prediction=str2num(rf_temp); %% here, RF's prediction needs cell2mat
    Dec_values{1,i}=dec_values;
    Prediction{1,i}=rf_prediction;
    accnum=0;
    for k=1:size(test1ab,1)
        %%%%%%%уш
        if(test1ab(k,1)==rf_prediction(k,1))
        accnum=accnum+1;
        end
    end
    accnum/size(test1ab,1)
    Accuracy{1,i}=accnum/size(test1ab,1);
end
end