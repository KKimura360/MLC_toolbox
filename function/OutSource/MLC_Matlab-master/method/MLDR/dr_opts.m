function opts = dr_opts(opts)
% function opts = dm_opts(opts)
% Options for dimensionality reduction package.
% This function checks the general settings of the input opts. 
% It also assigns all default values to empty fields, and checks possible errors for assigned fields.
% Note: it does not handle settings for specific algorithm. For example,
% the Laplacian type in HSL is not dealt with in this function. 
%
% Table of Options.  * * indicates default value.
%
%%  Part 1. Fields applicable for all dimensionality reduction algorithms.
%  FIELD                  DESCRIPTION
%% Algorithm Type:
% We provide an identifier to identify all supported dimensionality reduction algorithms
% .type                     **'cca': Canonical Correlation Analysis (CCA)
%                                  'lda': Linear Discriminant Analysis (LDA)
%                                  'opls': Partial Least Squares (PLS)
%                                  'hsl': Hypergraph Spectral Learning (HSL)
%% Output Format Specifications
% .k                          The number of projection vectors returned. 
%                              The default value is the total number of eigenectors corresponding to nonzero eigenvalues.
%                              And we use -1 to denote this default value. 
%% Regularization Parameter
% .reg_eig               Regularizaton parameter for the generalized eigenvalue problem
%                               **0 (reg_eig is not applicable for CCA)
% .reg_1norm          Regularization parameter for 1-norm
%                               **0
% .reg_2norm          Regularization parameter for 2-norm
%                               **0
%% Normalization Input
%. 
%% Implementation Methods
% .alg                       ** 'eig': the projection matrix is computed by solving the eigenvalue problem
%                                   'ls':   the projection matrix is computed by solving the corresponding least squares problem
%                                   'ls-lsqr': the projection matrix is computed by solving the corresponding least squares problem using lsqr algorithm
%                                   '2s': the projection matrix is computed by using the 2-stage algorithm
%                                   '2s-lsqr': the projection matrix computed by using the 2-stage algorithm, and the least squares problem
%                                                  in the 2nd stage is solved using the lsqr algorithm. 
% .lsqr_maxiter           The maximum iteration number in lsqr algorithm. The default value is 1e4.
% .lsqr_tol                    The tolerance in lsqr algorithm. The default value is 1e-6. 
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

%% Number of Outputs
if isfield(opts, 'k')
    if opts.k < 0
        opts.k = -1;
    end
else
    opts.k = -1;
end

%% Regularization Parameter
if isfield(opts, 'reg_eig')
    if opts.reg_eig < 0
        opts.reg_eig = 0;
    end
else
    opts.reg_eig = 0;
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


%% Implementation Methods
if isfield(opts, 'alg')
    if (strcmpi(opts.alg, 'eig')==0) && (strcmpi(opts.alg, 'ls')==0) && (strcmpi(opts.alg, '2s')==0 ...
            && (strcmpi(opts.alg, 'ls-lsqr')==0) && (strcmpi(opts.alg, '2s-lsqr')==0) )
        % defaul implementation method is by solving the eigenvalue problem.
        opts.alg = 'eig';
    end
else
    opts.alg = 'eig';
end

%% LSQR Algorithm Parameters
if isfield(opts, 'lsqr_maxiter') 
    if opts.lsqr_maxiter < 0
        opts.lsqr_maxiter = 1e4;
    end
elseif (strcmpi(opts.alg, 'ls-lsqr')==1) || (strcmpi(opts.alg, '2s-lsqr'))
    % the default iteration number is 1e4
    opts.lsqr_maxiter = 1e4;
end
if isfield(opts, 'lsqr_tol') 
    if opts.lsqr_tol < 0
        opts.lsqr_tol = 1e-6;
    end
elseif (strcmpi(opts.alg, 'ls-lsqr')==1) || (strcmpi(opts.alg, '2s-lsqr'))
    % the default iteration number is 1e4
    opts.lsqr_tol = 1e-6;
end




