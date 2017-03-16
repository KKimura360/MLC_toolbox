function[param]=SetRFSParameter(~)
%setREADERParameter

% opts.alpha The factor on the feature coefficient matrix
% opts.MaxIt The maximum number of iterations
% opts.dim   The dimensionality of feature subspace

param.alpha = 1;
param.iter  = 20;
param.dim   = '0.8*numF';
