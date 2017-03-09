function[param]=SetMLLEMParameter(pseudo)
%method.param{x}.
%dim:  reduced dim
%opsWI: 1,2 to obtain Instance-Instance relationship
%opsWL: 1,2,3 to obtain Label-Label relationship
%type : 'L' or 'NL', to map test instances
%k1   : number of k-nn for Instance-Instance
%k2   : number of k-nn for Label-Label
%k3   : number of k-nn for Nonlinear simulation (only for NL)
%SCtype= 1,2,3  Laplacian type, unnormalized, normalized by shi, normalized
%by jordan
%alpha: weighting parameter for Instance-Instance
%beta : wrighting paramter for Label-Label
%lambda: ridge paramter for Linear simulation (only for L) 

param.dim=20;
param.opsWI=1;
param.opsWL=2;
param.type='L';

param.k1=5;
param.k2=5;
param.k3=param.k2;
param.SCtype=1;

param.alpha=1;
param.beta=1;
param.lambda=10;


