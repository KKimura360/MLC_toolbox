function[model,method,time]=ridge_train(X,y,method)
%% Input
%X: Feature matrix (NxF)
%y: label vector
%NOTE: Unlike the the other methods, ridge regression is not allowed a class
%vector thus label vetor must be binary (0,1) or(-1,1) 
%in this implementation, to enjoy the sparsity of label vector, we use (0,1)  
%method.base.invX is common matrix for ridge regression

%if invX is already calculated by the others, use that 
if isfield(method.base,'invX')
    time=cputime;
    model=method.base.invX*y;
    time=cputime-time;
    return;
end

%if ridge parameter is not set
if ~isfield(method.base.param,'lambda')
    warning('ridge parameter is not set, default setting is applied l=0,1\n');
    lambda=0.1;
else
    lambda=method.base.param.lambda;
end


% obtain size 
[numN,numF]=size(X);

time=cputime;
%calculate weight matrix 
XX=[ones(numN,1),X];
invX=inv(XX' * XX + lambda * eye(size(XX, 2))) * XX';
%obtain weight matrix for label vector y
model=invX*y;
%to share the invX with the other labels (for BR)
method.base.invX=invX;
time=cputime-time;

    

 