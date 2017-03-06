function getBarPlot(Result,dataNames,functionNames,criterion,filename)
%% Input
%Result: cell(numData,numTrial,numCV,numMethod)
%dataNames: cell contains dataset names
%functionNames: cell contains names of methods for legend
%criterion: evaluation measurement to show
%% Output
% filename: to save 
%% Option
%filename to save the plot
[numData]=length(Result);
[numMethod]=length(Result{1});
[numTrial]=length(Result{1}{1});
[numCV]=length(Result{1}{1}{1}{1});
meanMat=zeros(numData,numMethod);
stdMat =zeros(numData,numMethod);

for i=1:numData
    dataname=dataNames{i};   
    tmpMat=zeros(numMethod,(numTrial*numCV));
    for j=1:numMethod
    for k=1:numTrial
    for l=1:numCV
        eval(['tmpMat(j,(k+(l-1)*k))=Result{i}{j}{k}{l}.',criterion,';']);
    end
    end
    end
    meanMat(i,:)=mean(tmpMat,2)';
    stdMat(i,:)=std(tmpMat');
end

%colorset
figure;
colormap(summer(numMethod));
tmpstd=zeros(numData,numMethod,2);
%lower
tmpstd(:,:,1)=stdMat;
tmpstd(:,:,2)=stdMat;
barwitherr(tmpstd,meanMat);

%set the other info
set(gca,'XTickLabel',dataNames,'XTick',1:numData);
title(criterion);
legend(functionNames,'Location','northwest');
grid on;

if nargin >4
    saveas(gcf,filename)
end    
    