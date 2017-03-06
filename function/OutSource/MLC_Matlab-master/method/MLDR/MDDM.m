function W= MDDM(X, Y, opts)
% function W= MDDM(X, Y, opts)
% MDDM: Multilabel-Dimensional reduction via Dependecies Maximization 
% Only the projection for X is computed.
%
%
% Usage:
%     W = MDDM(X, Y, opts)
%     W = MDDM(X, Y)
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point. (d-by-n matrix)
%        Y       - Data matrix Y. Each column of Y is a data point. (k-by-n matrix)
%                 - n is number of samples, d is the X data dimensionality, and k is the Y data dimensionality.
%       
%    Output:
%        W: each column is a projection vector for X. (d-by-k matrix)
% 
%    Examples:
%        X = rand(15,10);
%        Y = rand(3, 10);
%        opts.reg_eig = 0.5;
%        W = OPLS(X, Y, opts);
% 
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%
% 

% Step 1. Check the input opts using dr_opts function
if (~exist('opts','var'))
    opts = [];
end
opts.type = 'mddm';
opts = dr_opts(opts);

W = [];
if size(X,2)~=size(Y,2)
    disp('The numbers of samples in X and Y are not equal!');
    return;
end

% Centering of X
X = colCenter(X);

% Step 2. Sovle OPLS using different implemetations. The default choice is solving the eigenvalue problem.
if  strcmpi(opts.alg, '2s') || strcmpi(opts.alg, '2s-lsqr')     
    W = TwoStage_DR(X, Y, opts);    
else
    % The default implementation is solving the resulting generalized eigenvalue problem        
    L= Y' * Y;
    D= (X  * L) * X';
    D= D + 1e-09 * eye(size(D,1));
    opts.k=size(Y,1);
    %opts.k=floor(size(X,1)*0.3);
    [W tmp] = eigs(D,opts.k,'lm');
end