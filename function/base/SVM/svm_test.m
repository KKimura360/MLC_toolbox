function[conf,method,time]=svm_test(X,Y,Xt,model,method)
%X, Y and param are not needed for here (for K-nn classifiers)
%Xt: Feature matrix of test instances (Nt x F)
%model: learned by linear_svm_train
%% Output
%conf: confidence value of test instances for the label (Nt x1 real-value vector)
%method: returns the same method
%time: computation time for the prediction
%% Reference
%Chang, C. C., & Lin, C. J. (2011). LIBSVM: a library for support vector machines. ACM Transactions on Intelligent Systems and Technology (TIST), 2(3), 27.
%https://www.csie.ntu.edu.tw/~cjlin/libsvm/


[numNt,~]=size(Xt);
%psuedo-true label (to use liblinear)
yp=zeros(numNt,1);
% call libsvm 
time=cputime;
[conf]=svmpredict(yp,Xt,model,'-q');
time=cputime-time;
