function[param]=SetRFSParameter(~)
%SetRFSParameter

% opts.alpha The factor on the feature coefficient matrix
% opts.MaxIt The maximum number of iterations
% opts.dim   The dimensionality of feature subspace

param.alpha = 1;
param.maxIt = 100;
param.epsIt = 1e-5;
param.dim   = 100;
