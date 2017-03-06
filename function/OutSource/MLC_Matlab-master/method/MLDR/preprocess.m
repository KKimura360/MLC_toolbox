function [X_tr_new, X_te_new] = preprocess(X_tr, X_te, method)
% [X_tr_new, X_te_new] = preprocess(X_tr, X_te, method)
% preprocess the data (such as centralization or adding column for bias (for ML-LS))
% X_tr: d-by-n1 matrix
% X_te: d-by-n2 matrix
% d is the data dimensionality
% n1 is the sample size in the training set
% n2 is the sample size in the test set.
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%
switch method
    case {'ML-LS'}
        % add the bias row for all samples in training and test sets.
        n = size(X_tr, 2);
        X_tr_new = [X_tr; ones(1, n)];
        n = size(X_te, 2);
        X_te_new = [X_te; ones(1, n)];        
    otherwise
        X_tr_new = X_tr;
        X_te_new = X_te;
end