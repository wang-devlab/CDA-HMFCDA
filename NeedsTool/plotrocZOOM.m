function [VAUC,X1,Y1,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES] = plotrocZOOM(test_lab, Src_scores, KFold)
    hold on;
    

    % Times New Roman
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultTextFontName', 'Times New Roman');
    
    colors = lines(KFold);
    
    for i = 1:KFold
        src_scores = Src_scores{1,i};
        test1ab = test_lab{i,1};
        [X1,Y1,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test1ab, src_scores(:,1), '1');
        VAUC(1,i) = AUC;
        plot(X1, Y1, 'Color', colors(i,:), 'LineWidth', 1.2);        
    end    
    
    grid on;
    
    
    legend_labels = arrayfun(@(x) sprintf('%dth fold', x), 1:KFold, 'UniformOutput', false);
    ll = legend(legend_labels, 'Location', 'best');
      
    grid off;
    set(gca, 'FontSize', 10);
    set(ll, 'FontSize', 10);

    
    xlabel('Specificity', 'FontName', 'Times New Roman','FontSize', 14);
    ylabel('Sensitivity', 'FontName', 'Times New Roman','FontSize', 14);
    
    meanAUC = mean(VAUC);
    text(0.25, 0.2, num2str(meanAUC, 'Average AUC = %.4f'), 'Fontsize', 14);
    set(gca, 'box', 'on');

    %%%%%%%%%%
    % »­¾ØÐÎ¿ò
    rectangle('Position', [0.006, 0.85, 0.13, 0.14], 'LineWidth', 0.9, 'LineStyle', '--');
    
    
    annotation('textarrow', [0.215 0.275], [0.86 0.8], 'FontSize', 14);
    
    
    h1 = axes('position', [0.295 0.395 0.39 0.42]);
    axis(h1);
    hold on;
    
    for i = 1:KFold
        src_scores = Src_scores{1,i};
        test1ab = test_lab{i,1};
        [X1,Y1,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test1ab, src_scores(:,1), '-1');
        plot(X1, Y1, 'Color', colors(i,:), 'LineWidth', 1.5);
    end
    xlim([0 0.15]);
    ylim([0.8 0.99]);
    box on;
    
    hold off;
end