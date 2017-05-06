function[conf,time]=CCA_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by MLHSL_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%% Reference (APA style from google scholar)
%Hotelling, H. (1936). Relations between two sets of variates. Biometrika, 28(3/4), 321-377.

%%% Method  

%% Initialization
U=model{2};
time=cell(2,1);
tmptime=cputime;

%% Feature projection
meanX = mean(X,1);
tmpX  = bsxfun(@minus,X,meanX);
tmpXt = bsxfun(@minus,Xt,meanX);
tmpX  = tmpX * U;
tmpXt = tmpXt * U;
time{end}=cputime-tmptime;

%% Testing
[conf,time{1}]=feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));