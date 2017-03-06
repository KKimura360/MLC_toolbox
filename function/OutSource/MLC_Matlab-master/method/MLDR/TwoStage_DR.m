function [W, W1, W2] = TwoStage_DR(X, Y, opts)
% function [W, W1, W2] = TwoStage_DR(X, Y, opts)
% This function provides the uniform interface for the two-stage approach
% for all dimensionality reduction algorithms.
%
% ============= Input Description ============
% X: d-by-n data matrix.
% Y: k-by-n label matrix. Y(i,j) is either 1 or 0.
% opts: please refer to dr_opts function. 
% ============= Output Description ============
% Two-stage approach
% 1. W, W1 and W2: W1 is the solution of the first stage, and W2 is the solution of the
% second stage. W = W1*W2 is the final solution.
%
%
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

% Step 0. Parameter checking
opts = dr_opts(opts);
W = []; W1 = []; W2 = [];
[d, n] = size(X);
if n ~= size(Y, 2)
    disp('The data matrix X is not consistent with the label matrix Y');
    return;
end

% Step 1. Compute matrix H for different dimensionality reduction algorithms.
if strcmpi(opts.type, 'hsl')
    H = Y_decompose(Y, opts.Lap_type);
elseif strcmpi(opts.type, 'mddm')
    H= colCenter(Y)';
else
    H = Y_decompose(Y, opts.type);
end

% Step 2. The core part. Solve the generalized eigenvalue problem using
% two-stage approach
% We consider two implementations:
% 1) using SVD of X
% 2) using iterative algorithm lsqr

% Step 2.1: Stage 1. Solve the least squares problem using regularization
if strcmpi(opts.alg, '2s-lsqr')        
    W1 = [];
    if opts.reg_2norm == 0
        for  i = 1:size(H, 2)
            w = lsqr(X', H(:, i), opts.lsqr_tol, opts.lsqr_maxiter);
            W1 = [W1, w];
        end
    else       
        B = sparse(1:d, 1:d, opts.reg_2norm);
        X_ex = [X'; B];        
        for i = 1:size(H, 2)
            [w,~] = lsqr(X_ex, [H(:, i); zeros(d, 1)], opts.lsqr_tol, opts.lsqr_maxiter);
            W1 = [W1, w];
        end
    end
else
    % the default choise is 2s
%     [U1, S1, V1] = svd(X, 'econ');
    [U1, S1, V1] = rsvd(X, 200);
    r1 = rank(S1);
    U1 = U1(:, 1:r1); S1 = S1(1:r1, 1:r1); V1 = V1(:, 1:r1);
    s1 = diag(S1);
    if opts.reg_2norm > 0
        W1 = U1 * diag(s1./(s1.^2 + opts.reg_2norm)) * V1' * H;
    else
        W1 = U1 * diag(1./s1) * V1' * H;
    end
end

% Step 2.2: Stage 2.Solve the resulting optimization problem by replacing X with
% X_hat
D = W1' * X * H;
[UD, SD, ~] = svd(D, 'econ');
%     rD = rank(SD, 1e-6);
rD = rank(D);
UD = UD(:, 1:rD); SD = SD(1:rD, 1:rD);
sD = diag(SD);
W2 = UD * diag(1./sqrt(sD));


W = W1 * W2;