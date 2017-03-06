function W = lasso_func(X, T, opts)
% function W = lasso_func(X, T, opts)
% lasso_func: This function solves the LASSO problem. Note that this function
% acts as an interface to call LASSO. The users can choose different
% implementations of LASSO if the input/output is consistent with this
% function. In this version, we use the LeastR function provided by the SLEP package
% (http://www.public.asu.edu/~jye02/Software/SLEP/, by Jun Liu, Shuiwang Ji and Jieping Ye).
%
%
% Usage:
%     W = lasso_func(X, T, opts)
%     It solves the following optimization problem:
%     min_W ||W' * X - T||_F^2 + opts.reg_1norm ||W||_1 + opts.reg_2norm ||W||_2^2,
%     where the 1-norm and 2-norm of W are defined for each column of W separately.
% 
%    Input:
%        X       Data matrix X. Each column of X is a data point.(d-by-n matrix)
%        T       Target matrix T. Each column of T is a data point. (k-by-n matrix)
%        opts.reg_2norm:    The 2-norm regularization parameter. The default value is 0.
%        opts.reg_1norm:    The 1-norm regularization parameter. The default value is 0.
%    Here n is the number of data points, d is the input data dimensionality, and k is the dimensionality of the target.
% 
%
%    Output:
%        W: d-by-k matrix, each column is a projection vector for X.
% 
%    Examples:
%        X = rand(15,10);
%        T = rand(4, 10);
%        opts.reg_1norm = 0.5;
%        W = lasso_func(X, T, opts);
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

% % Step 1. Parameter checking for regularization.
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

if size(X, 2) ~= size(T, 2)
    disp('The numbers of samples in X and T are not equal!');
    return;    
end
[d, n] = size(X);
k = size(T, 1);

if issparse(X)
    X = full(X);
end
if issparse(T)
    T = full(T);
end

W = zeros(d, k);
for c = 1:k
    t = T(c, :)';
    % ** Users can change this line to use different implementations of lasso **
    % w = l1_ls(X', t, RegX, 1e-4, true);
    LeastR_opts.rsL2 = opts.reg_2norm;
    LeastR_opts.rFlag = 0;
    w = LeastR(X', t, opts.reg_1norm, LeastR_opts);
    % ** Users can change this line to use different implementations of lasso **
    W(:, c) = w;
end

