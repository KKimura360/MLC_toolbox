function[param]=SetMIFSParameter(~)
%setMIFSParameter

% opts.alpha The factor on the feature coefficient matrix
% opts.MaxIt The maximum number of iterations
% opts.dim   The dimensionality of feature subspace

param.alpha = 1;
param.beta  = 0.1;
param.gamma = 1;
param.dim   = 100;
param.k     = 0.3;
param.maxIt = 500;
opt_w.k     = 5;
opt_w.t     = 1;
opt_w.NeighborMode = 'KNN';
opt_w.WeightMode = 'HeatKernel';
param.opt_w = opt_w;
param.epsIt = 1e-3;
param.lambda = 1e-5;
