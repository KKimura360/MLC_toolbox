function[model,time]=CPLST_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.lambda: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%method.base.param= parameters of the base classifier
%% Output
%model: A learned model (cell(dim+2,1))
%model{1:dim}: classifier (regression) for latent labels
%model{dim+1}: Z latent labels
%model{dim+2}: Vm the other side matrix 
%% Reference (APA style from google scholar)
% Chen, Y. N., & Lin, H. T. (2012). Feature-aware label space dimension reduction for multi-label classification. In Advances in Neural Information Processing Systems (pp. 1529-1537).
% https://github.com/hsuantien/mlc_lsdr

%%% Method
% error check

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
% reduced dim (number of latent labels)
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end
% is a weighting paramter for reproductablity term see the paper 
lambda=method.param{1}.lambda;
model=cell(dim+3,1);
time=cell(dim+1,1);
tmptime=cputime;

%% Leaning model
%for the ridge regression
XX=[ones(numN,1),X];
invX=inv(XX' * XX + lambda * eye(size(XX, 2))) * XX';
H=XX * invX;
% to cnterize label matrix
shift = mean(Y);
Yshift = bsxfun(@minus,Y,shift);
[~, ~, V] = svd(Yshift' * H * Yshift, 0);
Vm = V(:, 1:dim);
Z = Yshift * Vm;

if strcmpi(method.base.name,'ridge')
    method.base.invX=invX;
end

time{end}=cputime-tmptime;
% CALL base classfier
for label=1:dim
    [model{label},method,time{label}]=feval([method.base.name,'_train'],X,Z(:,label),method);
end
% to keep necessary information
model{end-2}=Z;
model{end-1}=Vm;
model{end}=shift;
if isfield(method.base,'invX')
    method.base=rmfield(method.base,'invX');
end
