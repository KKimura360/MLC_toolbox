function[conf,time]=CPLST_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by CPLST_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%% Reference (APA style from google scholar)
% Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine learning, 85(3), 333.

%% Method 

%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
dim=method.param{1}.dim;
time=cell(1,1);
time{end}=0;

W=model{1};
H=model{2};
% Classify
conf=(H'*(W*Xt'));
conf=conf';
tmptime=cputime;
time{end}=cputime-tmptime;
