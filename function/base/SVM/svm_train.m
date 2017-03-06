function[model,method,time]=svm_train(X,y,method)
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
%Chang, C. C., & Lin, C. J. (2011). LIBSVM: a library for support vector machines. ACM Transactions on Intelligent Systems and Technology (TIST), 2(3), 27.
%https://www.csie.ntu.edu.tw/~cjlin/libsvm/

%if ridge parameter is not set
if ~isfield(method.base.param,'svmparam')
    warning('svm parameter is not set, default setting is applied (svm)\n');
    method.base.param.svmparam='-t 2';  %RBF-SVM
end
time=cputime;
model=svmtrain(y,X,method.svmparam);
time=cputime-time;


    

 