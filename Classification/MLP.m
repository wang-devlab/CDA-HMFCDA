function [Prediction, Accuracy, Dec_values] = MLP(test_data, test_lab, train_data, train_lab, KFold)
    % 初始化输出变量
    Dec_values = cell(1, KFold);
    Prediction = cell(1, KFold);
    Accuracy = zeros(1, KFold);
    
    
    for i = 1:KFold
        fprintf('Processing fold %d/%d...\n', i, KFold);
        
        
        testdata = test_data{i, 1};
        testlab = test_lab{i, 1};
        traindata = train_data{i, 1};
        trainlab = train_lab{i, 1};
        
        
        if ischar(trainlab) || iscellstr(trainlab)
            trainlab = str2double(trainlab);
            testlab = str2double(testlab);
        elseif iscategorical(trainlab)
            trainlab = double(trainlab);
            testlab = double(testlab);
            
            unique_labels = unique(trainlab);
            if length(unique_labels) == 2
                
                trainlab(trainlab == max(unique_labels)) = 1;
                trainlab(trainlab == min(unique_labels)) = -1;
                testlab(testlab == max(unique_labels)) = 1;
                testlab(testlab == min(unique_labels)) = -1;
            end
        end
        
        
        if ~isnumeric(trainlab)
            error('lable must be 1 or -1');
        end
        
        unique_labels = unique([trainlab; testlab]);
        if length(unique_labels) ~= 2
            error('only 2 classfication');
        end
        
        
        trainlab_cat = categorical(trainlab, unique_labels, ...
                                  cellstr(string(unique_labels)));
        testlab_cat = categorical(testlab, unique_labels, ...
                                 cellstr(string(unique_labels)));
        
        
        numFeatures = size(traindata, 2);
        numClasses = length(unique_labels);
        
        layers = [
            featureInputLayer(numFeatures, 'Name', 'input')
            fullyConnectedLayer(128, 'Name', 'fc1')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')
            dropoutLayer(0.3, 'Name', 'dropout1')
            fullyConnectedLayer(64, 'Name', 'fc2')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')
            dropoutLayer(0.3, 'Name', 'dropout2')
            fullyConnectedLayer(numClasses, 'Name', 'fc3')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];
        
        
        options = trainingOptions('adam', ...
            'MaxEpochs', 100, ...  
            'MiniBatchSize', 32, ...
            'Shuffle', 'every-epoch', ...
            'InitialLearnRate', 0.002, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, ...
            'LearnRateDropPeriod', 15, ...
            'Verbose', false, ...
            'Plots', 'none');
        
        
        rng(1); 
        net = trainNetwork(traindata, trainlab_cat, layers, options);
        
        
        [YPred_cat, scores] = classify(net, testdata);
        
       
        YPred = ones(size(YPred_cat));
        YPred(YPred_cat == categorical(unique_labels(1))) = -1;
        
        
        Prediction{1, i} = YPred;
        
        
        if numClasses == 2
            
            if unique_labels(1) > unique_labels(2)
                Dec_values{1, i} = scores(:, 1); 
            else
                Dec_values{1, i} = scores(:, 2); 
            end
        else
            Dec_values{1, i} = scores;
        end
        
        
        accnum = sum(YPred == testlab);
        Accuracy(1, i) = accnum / numel(testlab);
        
        fprintf('Fold %d accuracy: %.2f%%\n', i, Accuracy(1, i)*100);
    end
end