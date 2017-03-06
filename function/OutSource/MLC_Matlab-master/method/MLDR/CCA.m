function [W_x, W_y, corr_list] = CCA(X, Y, opts)
% function [W_x, W_y, corr_list] = CCA(X, Y, opts)
% CCA: Canonical Correlation Analysis
%
%
% Usage:
%     [W_x, W_y] = CCA(X, Y, options)
%     [W_x, W_y] = CCA(X, Y)
%
%    Input:
%        X     - Data matrix X. Each column of X is a data point.
%        Y      - Data matrix Y. Each column of Y is a data point.
%
%  Options specific for CCA:
%
%        opts.PrjX    = 1: The projection for X is requried.
%                             = 0: The projeciton for X is not required.
%                                    The default value is 1 (required).
%        opts.PrjY    = 1: The projection for Y is requried.
%                             = 0: The projeciton for Y is not required.
%                                    The default value is 1 (required).
%        opts.RegX  - The regularization parameter for X if opts.alg='eig'. The default value is 0.
%        opts.RegY  - The regularization parameter for Y if opts.alg='eig'. The default value is 0.
%
%
%    Output:
%        W_x: each column is a projection vector for X. If the projection
%                 for X is not required, an empty matrix is returned.
%        W_y: each column is a projection vector for Y. If the projection
%                 for Y is not required, an empty matrix is returned.
%        corr_list: the list of correlation coefficients in the projected spaces of X and Y.
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
% 

% Step 1. Check the input opts using dr_opts function
if (~exist('opts','var'))
    opts = [];
end
opts.type = 'cca';
opts = dr_opts(opts);

% Parameter checking specifically for HSL
% Default setting:
% If the algorithm is solving the eigenvalue problem, then we compute projections for both X and Y.
% Otherwise we compute the projection for X only.
if ~isfield(opts, 'PrjX')
    opts.PrjX = 1;
end
if ~isfield(opts, 'PrjY')
    if strcmpi(opts.alg, 'eig')
        opts.PrjY = 1;
    else
        opts.PrjY = 0;
    end
end

% Check other conflicting settings
% Standard:
% if alg is eig, then we support the following (PrjX, PrjY) pairs:
% (1,1), (1,0), (0,1).
% if alg is ls/ls-lsqr or 2s/2s-lsqr, we only support (1,0).
% For (0,0) pair, we convert it to the default setting.
% For (0,1) pair, we display an error message and return for ls/ls-lsqr and 2s/2s-lsqr alg.
% For (1,1) pair, we convert it to (1,0) for ls/ls-lsqr and 2s/2s-lsqr.
W_x = [];
W_y = [];
corr_list = [];

if opts.PrjX==1 && opts.PrjY==1
    if ~strcmpi(opts.alg, 'eig')
        opts.PrjY = 0;
        display('Projection for Y is not supported when algorithm is ls or 2s');
    end
end

if opts.PrjX==0 && opts.PrjY==1
    if ~strcmpi(opts.alg, 'eig')
        display('If one projection is computed for ls/2s/ls-lsqr/2s-lsqr, only the projection for X is supported');
        return;
    end
end

% convert to default setting is (PrjX, PrjY) is (0,0).
if opts.PrjX==0 && opts.PrjY==0
    if strcmpi(opts.alg, 'eig')
        opts.PrjX = 1;
        opts.PrjY = 1;
    else
        opts.PrjX = 1;
        opts.PrjY = 0;
    end
end

% set default value for regX and regY
if isfield(opts, 'reg_eig')
    if opts.reg_eig>0
        display('reg_eig is not supported by CCA. regX and regY are supported by CCA');
    end
end

if strcmpi(opts.alg, 'eig')
    if ~isfield(opts, 'regX')
        opts.regX = 0;
    elseif opts.regX<0
        opts.regX = 0;
    end
    if ~isfield(opts, 'regY')
        opts.regY = 0;
    elseif opts.regY<0
        opts.regY = 0;
    end
else
    % for ls and 2s, only reg_1norm and reg_2norm is considered for X.
    if isfield(opts, 'regY')
        if opts.regY>0
            display('only reg_1norm and reg_2norm are supported by ls and 2s in CCA');
        end
    end
    if isfield(opts, 'regX')
        if opts.regX>0
            display('only reg_1norm and reg_2norm are supported by ls and 2s in CCA');
        end
    end
    if isfield(opts, 'reg_eig')
        if opts.reg_eig>0
            display('only reg_1norm and reg_2norm are supported by ls and 2s in CCA');
        end
    end            
end

% Preprocess the input
if size(X, 2) ~= size(Y, 2)
    disp('The numbers of samples in X and Y are not equal!');
    return;
end
if issparse(X)
    X = full(X);
end
if issparse(Y)
    Y = full(Y);
end

% Center X and Y
X = colCenter(X);
Y = colCenter(Y);

% Compute the projections for different algorithms

%============ Core Algorithm =========
if strcmpi(opts.alg, 'ls') ||  strcmpi(opts.alg, 'ls-lsqr')
    if opts.PrjX==1        
        H = Y_decompose(Y, opts.type);
        [Q, R] = qr(H, 0);
        r = rank(H);
        Q = Q(:, 1:r);
        R = R(1:r, :);
        [UR, SR, VR] = svd(R);
        clear SR VR;
        UR = UR(:, 1:r);
        Target = Q * UR;
        W_x = solve_ls(X, Target', opts);    
    end
elseif strcmpi(opts.alg, '2s') || strcmpi(opts.alg, '2s-lsqr')
    if opts.PrjX==1
        W_x = TwoStage_DR(X, Y, opts);
    end
else 
    % The default implementation is solving the resulting generalized eigenvalue problem
    [Y_U, Y_Sigma, Y_V] = svd(Y, 'econ');
    rank_Y = rank(Y_Sigma);
    Y_U = Y_U(:, 1:rank_Y);
    Y_Sigma = Y_Sigma(1:rank_Y, 1:rank_Y);
    Y_sigma = diag(Y_Sigma);
    Y_V = Y_V(:, 1:rank_Y);
    
    % Construct matrix H, then we can call the general module to solve this
    % eigenvalue problem.
    if opts.PrjX==1
        if opts.regY==0
            % No regularization for Y is considered
            H = Y_V * Y_U';
        else
            % We consider the regularization for Y
            Y_sigma_reg = Y_sigma.^2 + opts.regY;
            Y_sigma_reg = sqrt(Y_sigma_reg);
            Y_sigma_reg = Y_sigma ./ Y_sigma_reg;
            H = Y_V * diag(Y_sigma_reg) * Y_U';
        end
        
        % Call the general function to solve the resulting eigenvalue problem
        [W_x, eigenV_list_x] = solve_eig(X, H, opts.regX);
        corr_list = sqrt(eigenV_list_x);
        
        % Compute the Projection for Y
        if opts.PrjY==1
            if opts.regY==0
                W_y = Y_U * diag(1./Y_sigma) * Y_V' * X' * W_x;
            else
                Y_sigma_reg = Y_sigma.^2 + opts.regY;
                Y_sigma_reg = Y_sigma ./ Y_sigma_reg;
                W_y = Y_U * diag(Y_sigma_reg) * Y_V' * X' * W_x;
            end
            % normalization
            for i=1:size(W_y,2)
                W_y(:,i) = W_y(:,i) / corr_list(i);
            end
        end
        
    else
        % If the projection for X is not required, we only focus on the
        % projection for Y
        if opts.PrjY==1
            % Compute the SVD of X
            [X_U, X_Sigma, X_V] = svd(X, 'econ');
            X_rank = rank(X_Sigma);
            X_U = X_U(:, 1:X_rank);
            X_Sigma = X_Sigma(1:X_rank, 1:X_rank);
            X_sigma = diag(X_Sigma);
            X_V = X_V(:, 1:X_rank);
            
            % Construct matrix H, then we can call the general module to solve the
            % eigenvalue problem.
            if opts.regX == 0
                % No regularization for Y is considered
                H = X_V * X_U';
            else
                % We consider the regularization for Y
                X_sigma_reg = X_sigma.^2 + opts.regX;
                X_sigma_reg = sqrt(X_sigma_reg);
                X_sigma_reg = X_sigma ./ X_sigma_reg;
                H = X_V * diag(X_sigma_reg) * X_U';
            end
            
            % Call the general function to solve the resulting eigenvalue problem
            [W_y, eigenV_list_y] = solve_eig(Y, H, opts.regY);
            corr_list = sqrt(eigenV_list_y);
        end
        
    end
end    

if opts.k>0
    if size(W_x, 2)>opts.k
        W_x = W_x(:, 1:opts.k);
    end
    if size(W_y, 2)>opts.k
        W_y = W_y(:, 1:opts.k);
    end
    if length(corr_list)>opts.k
        corr_list = corr_list(1:opts.k);
    end
end

