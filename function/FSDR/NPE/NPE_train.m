function[model,time]=NPE_train(X,Y,method)
%NPE: Neighborhood Preserving Embedding
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
%He, X., Cai, D., Yan, S., & Zhang, H. J. (2005, October). Neighborhood preserving embedding. In Computer Vision, 2005. ICCV 2005. Tenth IEEE International Conference on (Vol. 2, pp. 1208-1213). IEEE.

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
M     = speye(numN) - W;
M     = M'*M;
tmpX  = bsxfun(@minus,X,mean(X,1));
Sxx   = tmpX' * tmpX;
A     = tmpX' * M * tmpX;
B     = Sxx + gamma.*speye(numF);
[U,~] = eigs(A,B,dim);

%% CALL base classfier
tmpX      = tmpX * U;
model{2}  = U;
time{end} = cputime-tmptime;
[model{1},time{1}] = feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));