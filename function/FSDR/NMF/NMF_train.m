function[model,time]=NMF_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.dim   : lower-dim of features
%method.param{x}.iter  : number of iterations of NMF 
%% Output
%model: A learned model (cell(3,1))
%% Reference (APA style from google scholar)
%Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In Advances in neural information processing systems (pp. 556-562).

%%% Method
[numN,numF]=size(X);
[~,numL]=size(Y);

if ~isfield(method.param{1},'iter')
    method.param{1}.iter=30;
end

%% Initialization
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end
iter=method.param{1}.iter;
model=cell(2,1);
time=cell(2,1);
tmptime=cputime;
%Learning model
[U,V]=NMF(X,dim,iter);
% CALL base classfier
model{2} = U;
model{3} = V;
time{end}=cputime-tmptime;
[model{1},time{1}] = feval([method.name{2},'_train'],U,Y,Popmethod(method));