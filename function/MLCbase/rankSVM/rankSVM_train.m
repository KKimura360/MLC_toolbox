function[model,time]=rankSVM_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.svmparam: 'RBF' or 'Poly','Linear';
%method.param{x}.cost    : cost of SVM;
%recommend to use a defult setting
%method.param{x}.lambda_tol: tolerance paramter for lambda
%method.param{x}.norm_tol:  tolerance value for difference between alpha(p+1) and alpha(p)
%method.param{x}.max_iter: number of iterations
%% Output
%model{1}: Weights
%model{2}: Bias
%model{3}: SVs
%model{4}: Weights_sizepre
%model{5}: Bias_sizepre
%model{6}: svm_used
%% Reference
%Elisseeff, A., & Weston, J. (2001, December). A kernel method for multi-labelled classification. In NIPS (Vol. 14, pp. 681-687).
%http://cse.seu.edu.cn/people/zhangml/Resources.htm

%% NOTE
% linear programming solver returns error

%error check
if ~isfield(method.param{1},'svmtype')
    warning('method.param.svmparam is not set, we set Linear');
    method.param{1}.svmtype='Linear';
    method.param{1}.svmpara=[];
end


[numN numF]=size(X);
[numNL,numL]=size(Y);


%size check
sizeCheck;

%initialization
svm.type=method.param{1}.svmtype;
svm.para=method.param{1}.svmpara;
cost=method.param{1}.cost;
lambda_tol=method.param{1}.lambda_tol;
norm_tol=method.param{1}.norm_tol;
max_iter=method.param{1}.max_iter;

model=cell(6,1);
%Learning model
time=cputime;
[model{1},model{2},model{3},model{4},model{5},model{6},~]= ...
    RankSVM_train_raw(X,Y',svm,cost,lambda_tol,norm_tol,max_iter);
time=cputime-time;

