function[model,time]=Topk_train(X,Y,method)
%% Input
%X
%Y
%param.numk: top-k label
%% Output
%return empty

%%% method 

[numN,~]=size(X);
[~,numL]=size(Y);
time=cputime;
numk=method.param{1}.numk;
numk=min(numk,numL);

[~, labelRank]=sort(sum(Y),'descend');
model=labelRank;
time=cputime-time;
    

    

 