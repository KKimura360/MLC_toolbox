function[conf,time]=OPLS_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by FaIE_train
%% Output
%conf: confidence values (Nt x L);
%time: computaiton time
%
%% Reference (APA style from google scholar)
% OPLS: Worsley, K. J., Poline, J. B., Friston, K. J., & Evans, A. C. (1997). Characterizing the response of PET and fMRI data using multivariate linear models. NeuroImage, 6(4), 305-319.

%%% Method
%% Initialization
time=cell(2,1);
tmptime=cputime;
%NOTE X must be processed correctly on previous layers
% to shift in the same manner
U      = model{2};
Xmean  = mean(X,1);
tmpX   = bsxfun(@minus,X,Xmean);
tmpXt  = bsxfun(@minus,Xt,Xmean);
tmpX   = sparse(tmpX * U);
tmpXt  = sparse(tmpXt * U);
time{end}=cputime-tmptime;

[conf,time{1}]=feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));


