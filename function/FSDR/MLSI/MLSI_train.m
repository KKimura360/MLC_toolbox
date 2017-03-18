function[model,time]=MLSI_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.beta    a weight parameter
%method.param{x}.dim     lower-dim of labels
%method.param{x}.lambda  regularization parameter
%% Output
%model: A learned model of MLSI

%% Reference (APA style from google scholar)
%Yu, K., Yu, S., & Tresp, V. (2005, August). Multi-label informed latent semantic indexing. In Proceedings of the 28th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 258-265). ACM.

%%% Method

%% Initialization
dim=method.param{1}.dim;
gamma=method.param{1}.gamma;
beta=method.param{1}.beta;
[~,numF] = size(X);
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end
if dim >= numF
    error('the number of dim is larger than original dim')
end
model=cell(2,1);
time=cell(2,1);
tmptime=cputime;

%% Learning model
tmpX  = bsxfun(@minus,X,mean(X,1));
tmpY  = bsxfun(@minus,Y,mean(Y,1));
Sxx   = tmpX' * tmpX;
Sxy   = tmpX' * tmpY;
A     = (1-beta).*Sxx*Sxx + beta.*Sxy*Sxy';
B     = Sxx + gamma.*speye(numF);
[U,~] = eigs(A,B,dim);

%% CALL base classfier
tmpX  = tmpX * U;
model{2}=U;
time{end}=cputime-tmptime;
[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));