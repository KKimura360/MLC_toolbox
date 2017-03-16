function[param]=SetMIFSParameter(~)
%setREADERParameter

% opts.alpha The factor on the feature coefficient matrix
% opts.MaxIt The maximum number of iterations
% opts.dim   The dimensionality of feature subspace

param.alpha = 1;
param.beta  = 0.1;
param.gamma = 1;
param.dim   = '0.8*numF';
param.k     = 0.4;
param.tmax  = 800;
opt_w.k     = 10;
opt_w.NeighborMode = 'KNN';
opt_w.WeightMode = 'HeatKernel';
param.opt_w = opt_w;
param.epsilon = 1e-8;
param.lambda  = 1e-5;
