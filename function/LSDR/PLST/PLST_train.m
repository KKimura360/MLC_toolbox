function[model,time]=PLST_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.dim   : lower-dim of labels
%method.base.param= parameters of the base classifier
%% Output
%model: A learned model (cell(dim+2,1))
%model{1:dim}: classifier (regression) for latent labels
%model{dim+1}: Z latent labels
%model{dim+2}: Vm the other side matrix 
%time: computation time
%% Reference (APA style from google scholar)
% Tai, F., & Lin, H. T. (2012). Multilabel classification with principal label space transformation. Neural Computation, 24(9), 2508-2542.
% https://github.com/hsuantien/mlc_lsdr

% Note
% when a clustering method has been done before this and some labels do not
% appears at divided problem, this function may return error at eigs.
% thus matrix must be shrinked. 

%%% Method
%error check 

%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim',';']);
    dim=ceil(dim);
end
model=cell(dim+3,1);
time=cell(dim+1,1);
tmptime=cputime;
%size check
sizeCheck;

%this method is based on https://github.com/hsuantien/mlc_lsdr

%Learning model
[Z, Vm, shift] = plst_encode(Y, dim);
% CALL base classfier
time{end}=cputime-tmptime;
for label=1:dim
    [model{label},method,time{label}]=feval([method.base.name,'_train'],X,Z(:,label),method);
end
model{dim+1}=Z;
model{dim+2}=Vm;
model{dim+3}=shift;
if isfield(method.base,'invX')
    method.base=rmfield(method.base,'invX');
end
