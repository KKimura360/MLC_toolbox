function getTable(Result,dataNames,functionNames,criterion,filename)

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

meanMat=meanMat';
stdMat=stdMat';


result2TexTable(meanMat,stdMat,functionNames,dataNames,filename)