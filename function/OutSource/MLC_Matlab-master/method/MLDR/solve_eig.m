function [W, eigenV_list] = solve_eig(X, H, lambda, k)
% function W =solve_eig(X, H, lambda)
% This function computes the principal eigenvectors for the following
% eigenvalue problem:
%        (XX^T + lambda I)^(-1) (X S X^T) w = gamma * w.
% where X = [x1, ..., xn], X is d-by-n matrix, and S=HH^T is an n-by-n positive
% semidefinite matrix.
% Note that S = HH^T, where H is n-by-k matrix and it is provided.
% lambda is the regularization parameter.
% ============= Input Description ============
% X: d-by-n matrix, d is data dimensionality, n is the sampe size.
% H: n-by-k matrix, and S = HH^T.
% lambda: a scalar, the regularization parameter. The default value is 0.
% k: the number projection vectors returned. The default value is the total number of
% eigenvectors corresponding to nonzero eigenvalues.
% ============= Output Description ============
% W: d-by-k matrix, and each column corresponds to a projection vector.
%
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%


% Step 1. Input parameter checking
if issparse(X)
    X = full(X);
end
% If no regularization parameter is provided, we assume that the
% regularization parameter is 0.
if nargin < 4
    k = -1;
end
if nargin < 3
    lambda = 0;
end

% Step 2. Preprocess and SVD of X
%X = colCenter(X);
[U_X, Sigma_X, V_X] = svd(X, 'econ');
r = rank(Sigma_X);
U_X = U_X(:, 1:r);
Sigma_X = Sigma_X(1:r, 1:r);
V_X = V_X(:, 1:r);
sigma_X = diag(Sigma_X);

% Step 3. Solve the eigenvalue problem
if lambda == 0
    % No regularization
    B = V_X' * H;
    [P, Sigma_B, Q_B] = svd(B, 'econ');
    P = P(:, 1:rank(Sigma_B));
    clear Q_B;
    W = U_X * diag(1./sigma_X) * P;
else
    % Considering regularization
    %B = diag((diag(Sigma_X).^2 + lambda).^-0.5) * Sigma_X * V_X' * H;
    %B = diag((sigma_X.^2 + lambda).^-0.5) * Sigma_X * V_X' * H;
    B = diag( (sigma_X.^2+lambda).^-0.5 .* sigma_X ) * V_X' * H;
    [P, Sigma_B, Q] = svd(B, 'econ');
    clear Q;
    W = U_X * diag((sigma_X.^2 + lambda).^-0.5) * P;
end
sigma_B = diag(Sigma_B);
eigenV_list = sigma_B.^2;

if k<=size(W,2) && k >= 1
    W = W(:,1:k);
    eigenV_list = eigenV_list(1:k);
end



