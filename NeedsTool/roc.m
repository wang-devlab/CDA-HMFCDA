function [VACC,VSN,VPE,VMCC,VF1] = roc( predict_labe,test_data_labe,KFlod )
%ROC Summary of this function goes here
%   Detailed explanation goes here
    VACC{1,KFlod}=[];
    VPE{1,KFlod}=[];
    VSN{1,KFlod}=[];
    VMCC{1,KFlod}=[];
for i=1:KFlod
    predict_label=predict_labe{1,i};
    test_data_label=test_data_labe{1,i};

    len=length(predict_label);
    TruePositive = 0;
    TrueNegative = 0;
    FalsePositive = 0;
    FalseNegative = 0;
    for k=1:len
        if test_data_label(k)==1 && predict_label(k)==1  %真阳性
            TruePositive = TruePositive +1;
        end
        if test_data_label(k)==-1 && predict_label(k)==-1 %真阴性
            TrueNegative = TrueNegative +1;
        end 
        if test_data_label(k)==-1 && predict_label(k)==1  %假阳性
            FalsePositive = FalsePositive +1;
        end

        if test_data_label(k)==1 && predict_label(k)==-1  %假阴性
            FalseNegative = FalseNegative +1;
        end
    end
    TruePositive
    TrueNegative
    FalsePositive
    FalseNegative
    ACC = (TruePositive+TrueNegative)./(TruePositive+TrueNegative+FalsePositive+FalseNegative);
    SN = TruePositive./(TruePositive+FalseNegative);
    PE=TruePositive./(TruePositive+FalsePositive);
    MCC= (TruePositive*TrueNegative+FalsePositive*FalseNegative)./sqrt(  (TruePositive+FalseNegative)...
        *(TrueNegative+FalsePositive)*(TruePositive+FalsePositive)*(TrueNegative+FalseNegative));
    F1 = (2*SN*PE)/(SN+PE);
    
    VACC{1,i}=ACC;
    VPE{1,i}=PE;
    VSN{1,i}=SN;
    VMCC{1,i}=MCC;
    VF1{1,i}=F1;
end

end

