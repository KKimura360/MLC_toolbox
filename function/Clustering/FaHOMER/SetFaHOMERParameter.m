function[param]=SetFaHOMERParameter(pseudo)
%SetHOMERParameter
%Number of Clusters
param.numCls=5;
%ridge parameter
param.lambda=1;
% weight parameter for instance balancing
param.alpha=1;
% weight paramter for label balancing
param.beta=100;

