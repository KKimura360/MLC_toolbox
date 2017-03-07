function[model,time]=PCA_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.dim   : lower-dim of features
%% Output
%model: A learned model (cell(3,1))

%% Reference (APA style from google scholar)
%Peason, K. (1901). On lines and planes of closest fit to systems of point in space. Philosophical Magazine, 2(11), 559-572.


%% Method
%error check 

%% initialization
dim=method.param{1}.dim;
model=cell(3,1);
time=cell(2,1);
tmptime=cputime;
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end

%% Centering instances
meanX = mean(X,1);
X = bsxfun(@minus,X,meanX);

%Learning model
[W,~] = eigs(X'*X,dim);

% CALL base classfier
X = X * W;
model{2} = W;
model{3} = meanX;
time{end}=cputime-tmptime;
[model{1},time{1}] = feval([method.name{2},'_train'],X,Y,Popmethod(method));