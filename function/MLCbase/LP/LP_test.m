function[conf,time]=LP_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%model learned by LP_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it. 

%% Method 

%error check 

[numN,numF]=size(X);
[numNL,numL]=size(Y);

%initialization
[numNt,~]=size(Xt);
conf=zeros(numNt,numL);
%give a model to base classifer (which comes from base classifier at
%training stage)
[tmpLabel,~,time]=feval([method.base.name,'_test'],X,Y,Xt,model{1},method);
%obtain multi-label classifition result with labelset
%CAUTION, DONOT use regression for base model, tmpLabel must be integer,
%smaller than the label distinct.
conf= model{2}(tmpLabel,:);
