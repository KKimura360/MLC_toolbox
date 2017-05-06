function[model,time]=MHSL_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.gamma:  regularization parameter
%method.param{x}.dim:    lower-dim of labels
%method.param{x}.opt_w:  parameters for building affinity matrix
%% Output
%model: A learned model (cell(dim+2,1))
%
%% Reference (APA style from google scholar)
%Sun, L., Ji, S., & Ye, J. (2008, August). Hypergraph spectral learning for multi-label classification. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 668-676). ACM.

%%% Method

%% Initialization
[numN,numF] = size(X);
dim   = method.param{1}.dim;
gamma = method.param{1}.gamma;
opt_w = method.param{1}.opt_w;
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
tmpX  = full(bsxfun(@minus,X,mean(X,1)));
W     = constructW(Y,opt_w);
D     = sparse(1:numN,1:numN,sum(W,1),numN,numN);
Sxx   = tmpX' * tmpX;
A     = tmpX' * (D.^.5*W*D.^.5) * tmpX;
B     = Sxx + gamma*speye(numF);
A     = max(A,A');
B     = max(B,B');
[U,~] = eigs(A,B,dim);
U     = bsxfun(@rdivide,U,sqrt(sum(U.^2,1)));

%% CALL base classfier
tmpX      = tmpX * U;
model{2}  = U;
time{end} = cputime-tmptime;
[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));