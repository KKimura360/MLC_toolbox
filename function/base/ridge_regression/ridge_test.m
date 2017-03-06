function[conf,method,time]=ridge_test(X,Y,Xt,model,method)
%X, Y and param are not needed for here (for K-nn classifiers)
%Xt: Feature matrix of test instances (Nt x F)
%model: learned by linear_svm_train
%% Output
%conf: confidence value of test instances for the label (Nt x1 real-value vector)
%method: returns the same param
%time: computation time for the prediction
[numNt,numF]=size(Xt);
time=cputime;
% multiply the weight matrix (model)
XXt=[ones(numNt,1),Xt];
conf=XXt*model;
time=cputime-time;
