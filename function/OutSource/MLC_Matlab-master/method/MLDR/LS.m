function W = LS(X, T, opts)
% function W = LS(X, T, opts)
% This function provides a unifined interface to least squares whose implemetations
% are based on SVD of X.
% It can handle 1-norm and 2-norm regularization.
% Notes: this function can only handle nonsparse matrix.
%
% Usage:
%     W = LS(X, T, ops)
%     W = LS(X, T)
%
%    Input:
%        X:      Data matrix X. Each column of X is a data poin. (d-by-n matrix)
%        T:       Target matrix T. Each column of T is a data point. (k-by-n matrix)
%
%        opts.reg_2norm:    The 2-norm regularization parameter. The default value is 0.
%        opts.reg_1norm:    The 1-norm regularization parameter. The default value is 0.
%
%
%
%    Output:
%        W: each column is a projection vector for X. (d-by-k matrix)
%
%    Examples:
%        X = rand(15,10);
%        T = rand(4, 10);
%        opts.reg_2norm = 0.5;
%        W_x = LS(X, T, opts);
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%


% Step 1. Parameter checking for regularization.
if (~exist('opts','var'))
    opts = [];
end

if isfield(opts, 'reg_1norm')
    if opts.reg_1norm < 0
        opts.reg_1norm = 0;
    end
else
    opts.reg_1norm = 0;
end

if isfield(opts, 'reg_2norm')
    if opts.reg_2norm < 0
        opts.reg_2norm = 0;
    end
else
    opts.reg_2norm = 0;
end

W = [];
if size(X,2)~=size(T,2)
    disp('The numbers of samples in X and T are not equal!');
    return;
end

if issparse(X)
    X = full(X);
end
if issparse(T)
    T = full(T);
end


% Step 2. Solve the least squares formulation
if opts.reg_1norm > 0
    % Lasso
    W = lasso_func(X, T, opts);
elseif opts.reg_2norm > 0
    % Ridge Regression
    [U, Sigma, V] = svd(X, 'econ');
    r = rank(Sigma);
    U1 = U(:, 1:r);
    V1 = V(:, 1:r);
    sigma_r = diag(Sigma(1:r, 1:r));
    D = sigma_r.^2 + opts.reg_2norm;
    D = sigma_r ./ D;
    W = U1 * diag(D) * V1' * T';
else
    % Default: No regularization
    [U, Sigma, V] = svd(X, 'econ');
    r = rank(Sigma);
    U1 = U(:, 1:r);
    V1 = V(:, 1:r);
    sigma_r = diag(Sigma(1:r, 1:r));
    W = U1 * diag(1./sigma_r) * V1' * T';
end