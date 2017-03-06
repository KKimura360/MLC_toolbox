function[model,time]=MLSI_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.beta: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%% Output
%model: A learned model (cell(dim+2,1))

%% Reference (APA style from google scholar)
%Yu, K., Yu, S., & Tresp, V. (2005, August). Multi-label informed latent semantic indexing. In Proceedings of the 28th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 258-265). ACM.

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
tmpMat=(1-beta) .* (X*X') + beta .*(Y*Y');
tmpMat= tmpMat+ eps.* eye(size(tmpMat,1));
tmpMat= (X'* inv(tmpMat) * X) + (eye(size(X,2)));
[L, W]=eigs((X'*X),tmpMat,dim);

% CALL base classfier
tmpX= (X* L * W.^(1/2));
model{2}=L;
model{3}=W;
time{end}=cputime-tmptime;

[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));


