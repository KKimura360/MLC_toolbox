function[param]=SetHOMERParameter(~)
%SetHOMERParameter

%Number of Clusters
param.numCls=3;
%Clustering method
param.ClsMethod='bkmeans';
% Spectral clustering
param.sim = [];
