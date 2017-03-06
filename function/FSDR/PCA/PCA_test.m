function[conf,time]=PCA_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by PCA_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.

%% Reference (APA style from google scholar)
%Yu, K., Yu, S., & Tresp, V. (2005, August). Multi-label informed latent semantic indexing. In Proceedings of the 28th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 258-265). ACM.

%% Method 
% Get learned model
time=cell(2,1);
W = model{2};
meanX = model{3};
tmptime=cputime;
% Data mapping 
X  = bsxfun(@minus,X,meanX);
Xt = bsxfun(@minus,Xt,meanX);
tmpX  = X*W;
tmpXt = Xt*W;
time{end}=cputime-tmptime;
% Testing
[conf,time{1}] = feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));