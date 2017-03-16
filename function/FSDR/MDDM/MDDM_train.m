function[model,time]=MDDM_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.dim   : lower-dim of labels
%% Output
%model: A learned model (cell(dim+2,1))

%% Reference (APA style from google scholar)
%Zhang, Y., & Zhou, Z. H. (2010). Multilabel dimensionality reduction via dependence maximization. ACM Transactions on Knowledge Discovery from Data (TKDD), 4(3), 14.

%%% Method

%% Initialization
dim=method.param{1}.dim;
numF = size(X,2);
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end
if dim >= numF
    error('the number of dim is larger than original dim')
end
model=cell(2,1);
time=cell(2,1);
tmptime=cputime;

%% Learning model
Xmean  = mean(X,1);
tmpX   = bsxfun(@minus,X,Xmean);
tmpY   = bsxfun(@minus,Y,mean(Y,1));
Sxy    = tmpX' * tmpY;
A      = Sxy * Sxy';
[U,~]  = eigs(A,dim);
   
%% CALL base classfier
tmpX   = tmpX*U;
model{2}=U;
time{end}=cputime-tmptime;

[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));


