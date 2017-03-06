function[model,time]=MLC_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%% Output
%model: Learned by called model

%error check 
if ~isfield(method,'name')
    error('Multi label classifier is not set\n')
end

if ~isfield(method,'count')
    method.count=0;
end

[numN numF]=size(X);
[numNL,numL]=size(Y);

%size check
sizeCheck;

% the depth of layer
fprintf('CALL: %s\n ', method.name{1});
% in some cases, method will be update thus, base classifier returns method 
% if this depth is a clustering method
[model,time]=feval([method.name{1},'_train'],X,Y,method);


    
    
