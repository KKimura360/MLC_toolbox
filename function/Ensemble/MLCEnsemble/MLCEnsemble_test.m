function[conf, time]=ECC_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%% Reference (APA style from google scholar)


%% initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
numM=method.param{1}.numM;
time=cell(numM+1,1);
time{end}=0;
conf=zeros(numNt,numL); 
sumLabels=zeros(numNt,numL);


for i=1:numM
   tmpmethod.name=method.param{1}.names{i};
    tmpmethod.param=method.param{1}.params{i};
    tmpmethod.base=method.param{1}.bases{i};
    tmpmethod.th=method.param{1}.ths{i};
    [tmpconf,time{i}]=feval([tmpmethod.name{1},'_test'],X,Y,Xt,model{i},tmpmethod);
    conf=conf+tmpconf;
end
% divide by the total ensemble to obtain ratio
conf=conf./ numM;




