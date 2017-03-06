function[conf,method,time]=linear_svm_test(X,Y,Xt,model,method)
%% Input
%X, Y and param are not needed for here (for K-nn classifiers)
%Xt: Feature matrix of test instances (Nt x F)
%model: learned by linear_svm_train
%% Output
%conf: confidence value of test instances for the label (Nt x1 real-value vector)
%param: returns the same param
%time: computation time for the prediction
%% Reference 
%Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008). LIBLINEAR: A library for large linear classification. Journal of machine learning research, 9(Aug), 1871-1874.
% https://www.csie.ntu.edu.tw/~cjlin/liblinear/

Xt=sparse(Xt);
[numNt,~]=size(Xt);
%psuedo-true label (to use liblinear)
yp=zeros(numNt,1);
time=cputime;
% '-q' means silent mode
[conf]=predict(yp,Xt,model,'-q');
time=cputime-time;
