function getHeatmap(Result,dataname,change,criterion)
%Result contains {param1}{param2}{numTrial}{numCV}
[numParam1]=length(Result);
[numParam2]=length(Result{1});
[numTrial]=length(Result{1}{1});
[numCV]=length(Result{1}{1}{1});

heatMat=zeros(numParam1,numParam2);
for i=1:numParam1
for j=1:numParam2
for k=1:numTrial
    for l=1:numCV
        eval(['heatMat(i,j)=heatMat(i,j)+','Result{i}{j}{k}{l}.',criterion,';']);
    end
end
end
end
heatMat=heatMat / (numTrial*numCV);

figure;
imagesc(heatMat)
colorbar;

