function W = solve_ls(X, T, opts)
% This function provides the unified framework to solve least squares problem, including ordinary least squares (OLS),
% ridge regression (RR), lasso, and elastic net. It also provides different implementations for OLS and RR, i.e., solving it
% via SVD and the iterative lsqr algorithm. 
% ============= Input Description ============
% X: d-by-n matrix, and each column is a data point
% T: k-by-n matrix, the target matrix
% n is the number of samples, d is data dimensionality and k is the number of labels (classes). 
% opts: please refer to dr_opts function. 
% ============= Output Description ===========
% W: d-by-k matrix.
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

% Step 1. Parameter checking
W = [];
[d, n] = size(X);
[k, n1] = size(T);
if n1 ~= n    
    disp('the data and the response are not compatible');
    return;
end

if isfield(opts, 'reg_1norm')
    if opts.reg_1norm>0 && strcmpi(opts.alg, 'ls-lsqr')
        display('1-norm regularization is not supported by lsqr');
        return;
    end
end
        

% Step 2. Solve the least squares problem using different implementations.
if strcmpi(opts.alg, 'ls-lsqr') 
    % lsqr implementations, and ony 2-norm regularization is handled.
    W = zeros(d, k);
    if opts.reg_2norm == 0
        for i = 1:k
            display('solve lsqr');
            w = lsqr(X', T(i,:)', opts.lsqr_tol, opts.lsqr_maxiter);
            W(:, i) = w;
        end
    else
        if issparse(X)
            B = sparse(1:d, 1:d, opts.reg_2norm);
        else
            B = opts.reg_2norm * eye(d);
        end
        X_ex = [X'; B];
        for i = 1:k
            display('solve lsqr');
            w = lsqr(X_ex, [T(i, :)'; zeros(d, 1)], opts.lsqr_tol, opts.lsqr_maxiter);
            W(:, i) = w;
        end
    end
else
    % The implemetation based on X's SVD is computed. 
    W = LS(X, T, opts);
end

