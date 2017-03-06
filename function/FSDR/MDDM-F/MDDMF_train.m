function[model,time]=MDDMF_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.beta: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%% Output
%model: A learned model (cell(dim+2,1))

%% Reference (APA style from google scholar)
%Zhang, Y., & Zhou, Z. H. (2010). Multilabel dimensionality reduction via dependence maximization. ACM Transactions on Knowledge Discovery from Data (TKDD), 4(3), 14.

%%% Method

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
%reduced dimension
dim=method.param{1}.dim;

if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end

if dim >= numF
    error('the number of dim is larger than original dim')
end
%beta is a weight paramter for label information see the paper
beta=method.param{1}.beta;

model=cell(3,1);
time=cell(2,1);
tmptime=cputime;

%Learning model
L= Y * Y';
D= (X'  * L) * X;
D= D + 1e-09 * eye(size(D,1));
[W tmp] = eigs(D,dim,'lm');
% CALL base classfier

tmpX= X*W;
model{2}=tmp;
model{3}=W;
time{end}=cputime-tmptime;

[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));


