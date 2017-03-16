function[conf,time]=NPE_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by LPP_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%
%% Reference (APA style from google scholar)
%He, X., Cai, D., Yan, S., & Zhang, H. J. (2005, October). Neighborhood preserving embedding. In Computer Vision, 2005. ICCV 2005. Tenth IEEE International Conference on (Vol. 2, pp. 1208-1213). IEEE.

%%% Method 

%% Get learned model
U = model{2};
time=cell(2,1);
tmptime=cputime;

%% Feature projection
Xmean  = mean(X,1);
tmpX   = bsxfun(@minus,X,Xmean);
tmpXt  = bsxfun(@minus,Xt,Xmean);
tmpX   = tmpX * U;
tmpXt  = tmpXt * U;
time{end}=cputime-tmptime;

%% Testing
[conf,time{1}] = feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));