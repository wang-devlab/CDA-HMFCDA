function [VAUC,X1,Y1,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES]=plotroc(test_lab,Src_scores,KFold)
    hold on
    char=['r';'g';'b';'y';'k';'m';'c'];
    for i=1:KFold
        src_scores=Src_scores{1,i};
        test1ab=test_lab{i,1};
        [X1,Y1,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test1ab,src_scores(:,1),'-1');
        VAUC(1,i)=AUC;
        plot(X1,Y1,char(i,1),'LineWidth',1.5);
    end 
    grid on;
    % ll=legend('1th fold','2th fold','3th fold','4th fold','5th fold','6th fold','7th fold');
    ll=legend('1th fold','2th fold','3th fold','4th fold','5th fold');
    xlabel('Specificity');ylabel('Sensitivity');
    grid off;
    set(get(gca,'XLabel'),'FontSize',18);
    set(get(gca,'YLabel'),'FontSize',18);
    set(gca,'FontSize',10);
    set(ll,'FontSize',10);
    meanAUC=mean(VAUC);
    % stdAUC=std(VAUC);
    text(0.3,0.08,num2str(meanAUC,'Average AUC =%.4f'),'Fontsize',12)
set(gca,'box','on')
end