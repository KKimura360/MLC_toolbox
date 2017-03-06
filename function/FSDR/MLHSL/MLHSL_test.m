function[conf,time]=MLHSL_test(X,Y,Xt,model,method)
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
%Sun, L., Ji, S., & Ye, J. (2008, August). Hypergraph spectral learning for multi-label classification. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 668-676). ACM.
%%% Method  
[numN,numF]=size(X);
[numNL,numL]=size(Y);


%% Initialization
[numNt,~]=size(Xt);
time=cell(2,1);
tmptime=cputime;
% obtain regression result
L=model{2};
W=model{3};
tmpX= X*W;
tmpXt=Xt*W;
time{end}=cputime-tmptime;
[conf, time{1}]=feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));


