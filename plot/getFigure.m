function getFigure(Result,dataname,change,criteria,filename)
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
[numCriteria]=length(criteria);
meanMat=zeros(numCriteria,numParam);
stdMat=meanMat;
for i=1:numCriteria;
    tmpvec=zeros(numParam,(numTrial*numCV));
    for j=1:numParam
    for k=1:numTrial
    for l=1:numCV
       eval(['tmpMat(j,(k+(l-1)*k))=Result{j}{k}{l}.',criteria{i},';']);
    end
    end
    end
meanMat(i,:)=mean(tmpMat,2);
stdMat(i,:)=std(tmpMat');
end

% Preliminary
Styles={'r-+','y-s','g-x','k-*','m-^','b-o'};
maxy=max(max(meanMat));
miny=min(min(meanMat));

figure;
ylim([miny, maxy]);
for i=1:numCriteria
    plot(1:numParam,meanMat(i,:),Styles{i},'LineWidth',2,'MarkerSize',7);
    hold on;
end
    set(gca,'XTickLabel',change.value,'XTick',1:numParam);
    xlabel([change.name,'-',change.param]);
    title(dataname); 
    grid on;
    legend(criteria);
