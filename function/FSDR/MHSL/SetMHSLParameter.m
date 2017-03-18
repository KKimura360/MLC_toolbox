function[param]=SetMHSLParameter(~)
%SetHOMERParameter

%Number of Clusters
param.gamma  = 1;
param.dim    = 100;
opt_w.k      = 10;
opt_w.NeighborMode = 'KNN';
opt_w.WeightMode   = 'HeatKernel';
param.opt_w  = opt_w;
