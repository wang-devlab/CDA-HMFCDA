function [Prediction, Accuracy, Dec_values] = AdaBoost(test_data, test_lab, train_data, train_lab, K)
    
    Prediction = cell(1, K);
    Accuracy = zeros(K, 1);
    Dec_values = cell(1, K);

    if length(test_data) ~= K || length(test_lab) ~= K || ...
       length(train_data) ~= K || length(train_lab) ~= K
        error('K error');
    end
    
    
    for fold = 1:K
        fprintf('Processing fold %d/%d...\n', fold, K);
        
        
        XTrain = train_data{fold};
        YTrain = train_lab{fold};
        XTest = test_data{fold};
        YTest = test_lab{fold};
        
        
        rng(1); 
        classifier = fitcensemble(XTrain, YTrain, ...
                                'Method', 'AdaBoostM1', ... 
                                'NumLearningCycles', 128, ...
                                'Learners', 'tree', ...
                                'ClassNames', [1; -1]);
        
        
        [YPred, scores] = predict(classifier, XTest);
        
        
        Prediction{fold} = YPred;
        Accuracy(fold) = sum(YPred == YTest) / length(YTest);
        Dec_values{fold} = scores(:,2); 
        
        fprintf('Fold %d accuracy: %.2f%%\n', fold, Accuracy(fold)*100);
    end
end