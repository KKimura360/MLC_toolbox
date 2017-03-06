function[model,time]=MLCCAF_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.beta: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%method.param{x}.type  : 
%% Output
%model: A learned model (cell(dim+2,1))
%% Reference (APA style from google scholar)
%Sun, L., Ji, S., & Ye, J. (2011). Canonical correlation analysis for multilabel classification: A least-squares formulation, extensions, and analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1), 194-200.

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

type=method.param{1}.type;
lambda=method.param{1}.lambda;
model=cell(3,1);
time=cell(2,1);
tmptime=cputime;

%Learning model
if issparse(X)
    X = full(X);
end
if issparse(Y)
    Y = full(Y);
end
X = colCenter(X');
Y = colCenter(Y');

[Y_U, Y_Sigma, Y_V] = svd(Y, 'econ');
rank_Y = rank(Y_Sigma);
Y_U = Y_U(:, 1:rank_Y);
Y_Sigma = Y_Sigma(1:rank_Y, 1:rank_Y);
Y_sigma = diag(Y_Sigma);
Y_V = Y_V(:, 1:rank_Y);
H = Y_V * Y_U';
% CALL base classfier
tmpX= X*W ;
model{3}=W;
time{end}=cputime-tmptime;

[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));


