function[param]=SetREADERParameter(~)
%setREADERParameter

% opts.alpha The factor on the feature coefficient matrix
% opts.beta  The factor on manifold learning of Xw
% opts.gamma The factor on manifold learning of Y
% opts.k     The dimensionality of embedded label space
% opts.p     The number of nearest neighbors
% opts.MaxIt The maximum number of iterations
% opts.dim   The dimensionality of feature subspace

param.alpha = 1;
param.beta  = 1;
param.gamma = 1;
param.k     = 1;
param.p     = 10;
param.MaxIt = 50;
param.dim   = 50;
