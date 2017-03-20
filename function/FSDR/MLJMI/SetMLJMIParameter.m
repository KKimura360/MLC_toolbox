function[param]=SetMLJMIParameter(~)
%SetMLJMIParameter
% param.dim     Dimensionality of feature subspace
% param.numStat Number of statuses of discretized features (3 or 5)
% param.factor  Factor of discretization
% param.maxF    Factor on limiting the search space

param.dim     = 100;
param.numStat = 3;
param.factor  = 1;
param.maxF    = 500;