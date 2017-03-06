function[conf,time]=MLCCAF_test(X,Y,Xt,model,method)
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
%Sun, L., Ji, S., & Ye, J. (2011). Canonical correlation analysis for multilabel classification: A least-squares formulation, extensions, and analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1), 194-200.

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


