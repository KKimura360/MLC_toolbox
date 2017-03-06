function W= LDA(X, Y, opts)
% function W= LDA(X, Y, opts)
% LDA: Linear Discriminant Analysis
% Only the projection for X is computed.
%
%
% Usage:
%     W = LDA(X, Y, opts)
%     W = LDA(X, Y)
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point. (d-by-n matrix)
%        Y       - Data matrix Y. Each column of Y is a data point. (k-by-n matrix)
%                 - n is number of samples, d is the X data dimensionality, and k is the Y data dimensionality.
%       
%
%    Output:
%        W: each column is a projection vector for X. If the projection
%                 for X is not required, an empty matrix is returned. (d-by-k matrix)
% 
%    Examples:
%        X = rand(15,10);
%        Y = rand_label(3, 10);
%        opts.reg_eig = 0.5;
%        W = LDA(X, Y, opts);
% 
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%
% 

% Step 1. Check the input opts using dr_opts function
if (~exist('opts','var'))
    opts = [];
end
opts.type = 'lda';
opts = dr_opts(opts);

W = [];
if size(X,2)~=size(Y,2)
    disp('The numbers of samples in X and Y are not equal!');
    return;
end

% Centering X
X = colCenter(X);


% Step 2. Sovle OPLS using different implemetations. The default choice is solving the eigenvalue problem.
if strcmpi(opts.alg, 'ls') || strcmpi(opts.alg, 'ls-lsqr')
    H = Y_decompose(Y, opts.type);
    [Q, R] = qr(H, 0);
    r = rank(H);
    Q = Q(:, 1:r);
    R = R(1:r, :);
    [UR, SR, VR] = svd(R);
    clear SR VR;
    UR = UR(:, 1:r);
    Target = Q * UR;
    W = solve_ls(X, Target', opts); 
elseif strcmpi(opts.alg, '2s') || strcmpi(opts.alg, '2s-lsqr')     
    W = TwoStage_DR(X, Y, opts);    
else
    % The default implementation is solving the resulting generalized eigenvalue problem        
    H = Y_decompose(Y, opts.type);
    W = solve_eig(X, H, opts.reg_eig, opts.k);
end


