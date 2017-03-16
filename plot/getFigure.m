function getFigure(Result,dataname,change,criterion,filename)
%% Input
%Result: cell(numData,numTrial,numCV,numMethod)
%dataNames: datasetname
%functionNames: cell contains names of methods for legend
%change: contains how parameters were changed
%    change.name=method.name 
%    change.param=paramter name
%    change.value=values
%criterion: evaluation measurement to show
%% Output
% filename: to save 
%% Option
%filename to save the plot

[numParam]=length(Result);
[numTrial]=length(Result{1});
[numCV]=length(Result{1}{1});


tmpMat=zeros(numParam,(numTrial*numCV));
for j=1:numParam
for k=1:numTrial
for l=1:numCV
    eval(['tmpMat(j,(k+(l-1)*k))=Result{j}{k}{l}.',criterion,';']);
end
end
end

meanVec=mean(tmpMat);


% Preliminary
Styles={'r-+'};
maxy=max(max(meanVec));
miny=min(min(meanVec));

figure;
ylim([miny, maxy]);
plot(1:numParam,meanVec,Styles{1},'LineWidth',2,'MarkerSize',7);
hold on;
set(gca,'XTickLabel',change.value,'XTick',1:numParam);
xlabel([change.name,'-',change.param]);
title([dataname,'-',criterion]); 
grid on;
        
if nargin >4
    saveas(gcf,filename)
end    
