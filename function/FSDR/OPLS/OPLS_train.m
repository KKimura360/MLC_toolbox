function[model,time]=OPLS_train(X,Y,method)
%
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.beta: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%% Output
%model: A learned model (cell(2,1))
%time : computation time
%
%% Reference (APA style from google scholar)
%Worsley, K. J., Poline, J. B., Friston, K. J., & Evans, A. C. (1997). Characterizing the response of PET and fMRI data using multivariate linear models. NeuroImage, 6(4), 305-319.

%%% Method

%% initialization
dim=method.param{1}.dim;
gamma=method.param{1}.gamma;
[~,numF] = size(X);
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end
if dim >= numF
    error('the number of dim is larger than original dim')
end
time=cell(2,1);
tmptime=cputime;
model=cell(2,1);

%% Solve the generalized eigenvalue problem: A*U = B*U*D 
tmpX  = full(bsxfun(@minus,X,mean(X,1)));
tmpY  = bsxfun(@minus,Y,mean(Y,1));
Sxx   = tmpX' * tmpX;
Sxy   = tmpX' * tmpY;
A     = Sxy * Sxy';
B     = Sxx + gamma*speye(numF);
A     = max(A,A');
B     = max(B,B');
[U,~] = eigs(A,B,dim);
U     = bsxfun(@rdivide,U,sqrt(sum(U.^2,1)));

%% Feature projection
tmpX  = tmpX * U;
model{2}=U;
time{end}=cputime-tmptime;

%CALL next Classifier
[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));