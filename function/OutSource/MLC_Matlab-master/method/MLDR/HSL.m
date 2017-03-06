function W = HSL(X, Y, opts)
% function W = HSL(X, Y, opts)
% This function provides the interfact to Hypergraph  Spectral Learning (HSL), including its different implementations.
%
% ============= Input Description ============
% X: d-by-n matrix, the data matrix.
% Y: k-by-n matrix, the label matrix. Y(i, j) is either 1 or 0.
%     d is the data dimensionality, k is the number of classes, and n is
%     the number of samples
% opts: please refer to dr_opts function. 
% ============= Output Description ============
% W: d-by-k matrix, and each column corresponds to a projection vector.
%
% Options specific for HSL:
% .Lap_type          **'clique': the clique expansion is used to compute the Laplacian matrix for a hypegraph.
%                               'star':     the star expansion  is used to compute the Laplacian matrix for a hypegraph
%                               'zhou':    Zhou's Laplacian is used to compute the Laplacian matrix for a hypergraph
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

% Step 1. Check the input opts using dr_opts function
if (~exist('opts','var'))
    opts = [];
end
opts.type = 'hsl';
opts = dr_opts(opts);

% Parameter checking specifically for HSL
if isfield(opts,'Lap_type')
    if (strcmpi(opts.Lap_type,  'clique')==0) && (strcmpi(opts.Lap_type, 'star')==0) && (strcmpi(opts.Lap_type, 'zhou')==0)
        opts.Lap_type = 'clique';
    end
else
    opts.Lap_type = 'clique';
end

W = [];
if size(X,2)~=size(Y,2)
    disp('The numbers of samples in X and Y are not equal!');
    return;
end

% Centering X
X = colCenter(X);

% Step 2. Sovle HSL using different implemetations. The default choice is solving the eigenvalue problem.
if strcmpi(opts.alg, 'ls') || strcmpi(opts.alg, 'ls-lsqr')
    H = Y_decompose(Y, opts.Lap_type);
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
    H = Y_decompose(Y, opts.Lap_type);
    W = solve_eig(X, H, opts.reg_eig, opts.k);
end