function[model,time]=LPP_train(X,Y,method)
%LPP: Locality Preserving Projections
%
% Solve the following optimization problem
% min_u u'X'LXu   s.t. u'X'D'Xu = 1
% which is equivalent to
% max_u u'X'WXu   s.t. u'X'D'Xu = 1
% since L = D - W
%
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.dim   : lower-dim of features
%% Output
%model: A learned model of PCA
%
%% Reference (APA style from google scholar)
% He, X., & Niyogi, P. (2003, December). Locality preserving projections. In NIPS (Vol. 16, No. 2003).

%%% Method

%% initialization
dim   = method.param{1}.dim;
gamma = method.param{1}.gamma;
opt_w = method.param{1}.opt_w;
[numN,numF] = size(X);
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end
if dim >= numF
    error('the number of dim is larger than original dim')
end
model   = cell(2,1);
time    = cell(2,1);
tmptime = cputime;

%% Learning model
W     = constructW(X,opt_w);
D     = sparse(1:numN,1:numN,sum(W,1),numN,numN);
tmpX  = bsxfun(@minus,X,mean(X,1));
A     = tmpX' * W * tmpX;
B     = tmpX' * D * tmpX + gamma.*speye(numF);
[U,~] = eigs(A,B,dim);

%% CALL base classfier
tmpX      = tmpX * U;
model{2}  = U;
time{end} = cputime-tmptime;
[model{1},time{1}] = feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));