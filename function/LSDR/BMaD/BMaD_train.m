function[model,time]=BMaD_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%param.dim: reduced dimension
%tau is threshold paramter for boolean matrix decompstion
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
if ~isfield(method.param{1},'tau');
    warning('parameter tau is not set\n we use alpha=0.1\n');
    method.param{1}.tau=0.5;
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
dim=min(dim,numL);
[Z V]=asso(Y,NumFactor,tau); % Z: Nxdim Y: KxL  

model=cell(3,1);
 [model{1},time{label}]=feval([method.name{2},'_train'],X,Z,Popmethod(method));
model{2}=Z;
model{3}=V;



