function[model,time]=CCA_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.gamma:  regularization parameter
%method.param{x}.dim:    lower-dim of labels
%% Output
%model: A learned model (cell(dim+2,1))
%% Reference (APA style from google scholar)
%Hotelling, H. (1936). Relations between two sets of variates. Biometrika, 28(3/4), 321-377.

%%% Method

%% Initialization
numF  = size(X,2);
dim   = method.param{1}.dim;
gamma = method.param{1}.gamma;
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
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
tmpY  = bsxfun(@minus,Y,mean(Y,1));
tmpY  = tmpY(:,any(tmpY,1)); 
Sxx   = tmpX' * tmpX;
Sxy   = tmpX' * tmpY;
Syy   = tmpY' * tmpY;
A     = Sxy / Syy * Sxy';
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