function[model,method,time]=linear_svm_train(X,y,method)
%% Input
%X: Feature matrix (NxF)
%y: label vector  (Nx1)
%NOTE: label vector can be used for class vector, in other words,
%multi-class classification is also allowed with this code.
%method.base.param.svmparam is option of liblinear, see their pages
%% Output
%model weight matrix
%method: returns the model back (for ridge regression)
%time returns computation time
%% Reference 
%Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008). LIBLINEAR: A library for large linear classification. Journal of machine learning research, 9(Aug), 1871-1874.
% https://www.csie.ntu.edu.tw/~cjlin/liblinear/

%if svm parameter is not set
if ~isfield(method.base.param,'svmparam')
    warning('svm parameter is not set, default setting is applied (linear svm)\n');
    method.base.param.svmparam='-s 2 -B 1 -q'; %L2^regularized L2-loss SVC regression
end

[numN,numF] = size(X);
if numN > numF
    method.base.param.svmparam='-s 2 -B 1 -q';  % Solve dual problem -- faster
else
    method.base.param.svmparam='-s 1 -B 1 -q';  % Solve primal problem -- faster
end

%train the model
X=sparse(X);
time=cputime;
model = train(y,X,method.base.param.svmparam);
time=cputime-time;




    

 