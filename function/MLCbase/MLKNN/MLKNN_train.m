function[model,time]=MLKNN_train(X,Y,method)
%% Input
%X: Feature matrix (NxF)
%y: label vector
%% Output
% return empty
%Reference

%%% method 
[numN,~]=size(X);
time=cputime;
type=method.param{1}.type;
numk=method.param{1}.numk;
numk=min(numk,numN);
switch type
    case 1
        %ML-zhang MLKNN
        model=cell(4,1);
        smooth=method.param{1}.smooth;
        [model{1},model{2},model{3},model{4}]=MLKNN_train_raw(X,Y',numk,smooth);
    case 2
        %SLEEC based simple k-nn 
        model=''; % nothing to learn here 
    otherwise
        error('type is wrong');
end
time=cputime-time;
    

    

 