function[model,time]=CSSP_train(X,Y,method)
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
%% Reference (APA style from google scholar)
%Hsu, D. J., Kakade, S., Langford, J., & Zhang, T. (2009, December). Multi-label prediction via compressed sensing. In NIPS (Vol. 22, pp. 772-780).
%https://github.com/hsuantien/mlc_lsdr

%%% Method

%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
% reduced dim of labels
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end

model=cell(dim+2,1);
time=cell(dim+1,1);
tmptime=cputime;


%Learning model
[Z, Vm] = cssp_encode(Y, dim);
% CALL base classfier
time{end}=tmptime-cputime;
for label=1:dim
    [model{label},method,time{label}]=feval([method.base.name,'_train'],X,Z(:,label),method);
end
model{dim+1}=Z;
model{dim+2}=Vm;

