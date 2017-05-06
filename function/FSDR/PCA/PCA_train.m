function[model,time]=PCA_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.dim   : lower-dim of features
%% Output
%model: A learned model of PCA

%% Reference (APA style from google scholar)
%Peason, K. (1901). On lines and planes of closest fit to systems of point in space. Philosophical Magazine, 2(11), 559-572.


%%% Method

%% initialization
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end
if dim >= size(X,2)
    error('the number of dim is larger than original dim')
end
model   = cell(2,1);
time    = cell(2,1);
tmptime = cputime;

%% Learning model
tmpX  = full(bsxfun(@minus,X,mean(X,1)));
Sxx   = tmpX'*tmpX;
A     = max(Sxx,Sxx');
[U,~] = eigs(A,dim);

%% CALL base classfier
tmpX      = tmpX * U;
model{2}  = U;
time{end} = cputime-tmptime;
[model{1},time{1}] = feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));