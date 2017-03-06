function[model,time]=FaIE_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.lambda: a weight parameter
%method.param{x}.alpha: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%method.base.param= parameters of the base classifier
%% Output
%model: A learned model (cell(dim+2,1))
%model{1:dim}: classifier (regression) for latent labels
%model{dim+1}: Z latent labels
%model{dim+2}: Vm the other side matrix 
%% Reference (APA style from google scholar)
% Lin, Z., Ding, G., Hu, M., & Wang, J. (2014). Multi-label Classification via Feature-aware Implicit Label Space Encoding. In ICML (pp. 325-333).
% https://github.com/hsuantien/mlc_lsdr

%% NOTE
%%% Method

%error check 
if ~strcmpi(method.base.name,'ridge');
    error('FaIE only considers ridge regression\n')
end

if ~isfield(method.param{1},'alpha');
    warning('parameter alpha is not set\n we use alpha=0.1\n');
    method.param{1}.alpha=0.1;
end

if ~isfield(method.param{1},'lambda')
     warning('parameter lambda is not set\n we use lambda=0.1\n');
    if ~isfield(method.base.param,'lambda');
        method.base.param.lambda=0.1;
    end
     method.param{1}.lambda=method.base.param.lambda;
end

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
% reduced dim of labels
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end
%alpha is weight for reprouctablity. see the paper. 
alpha=method.param{1}.alpha;
%lambda is the ridge parameter 
lambda=method.param{1}.lambda;
model=cell(dim+2,1);
time=cell(dim+1,1);
tmptime=cputime;
%% Learning model
% ridge regression part
XX=[ones(numN,1),X];
method.base.invX=inv(XX' * XX + lambda * eye(size(XX, 2))) * XX';
H=XX* method.base.invX;
%Label space dimension reduction
Omega = Y * Y' + alpha * H;
[Z, D] = eigs(Omega, dim);
Vm = (Z' * Y)';
%% CALL base classfier (should be ridge regression)
time{end}=cputime-tmptime;
for label=1:dim
    [model{label},method,time{label}]=feval([method.base.name,'_train'],X,Z(:,label),method);
end
model{dim+1}=Z;
model{dim+2}=Vm;
if isfield(method.base,'invX')
    method.base=rmfield(method.base,'invX');
end
