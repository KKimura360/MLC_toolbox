function[conf,time]=LPP_test(X,Y,Xt,model,method)
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
%He, X., & Niyogi, P. (2003, December). Locality preserving projections. In NIPS (Vol. 16, No. 2003).

%%% Method 

%% Get learned model
U = model{2};
time=cell(2,1);
tmptime=cputime;

%% Feature projection
X   = X * U;
Xt  = Xt * U;
time{end}=cputime-tmptime;

%% Testing
[conf,time{1}] = feval([method.name{2},'_test'],X,Y,Xt,model{1},Popmethod(method));
