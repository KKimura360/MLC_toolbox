function[model,time]=BR_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.base.name= base classifier
%method.base.param= parameters of the base classifier
%% Output
%model: weight matrix
% time: computation time
%%% method

%% Initialization
[numN, numF]=size(X);
[numNL,numL]=size(Y);

model=cell(numL,1);
time=cell(numL+1,1);
time{end}=0;

%Learning model
% fprintf('CALL: %s\n',method.base.name);
for label=1:numL
    [model{label},method,time{label}]=feval([method.base.name,'_train'],X,Y(:,label),method);
end


