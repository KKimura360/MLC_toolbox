function[param]=SetCBMLCParameter(pseudo)
%SetHOMERParameter

%Number of Clusters
param.numCls=5;
%Clustering method
param.ClsMethod='hkmeans';

%param.ClsMethod='SC';
%param.sim.type='Lab-nn';
%param.sim.k=5;
%param.SCtype=1;

%for hierarchical k-means
param.mxPts=1000;