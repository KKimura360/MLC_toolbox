
function [data,target,indices]=read_Data(dataname,numCV,seed)
%% Input
% dataname: name of dataset (call dataname.mat from dataset/matfile/
% numCV   : the number of separation of data
%% Output
% data    : feature matrix (N x F matrix)
% target  : label   matrix (L x N matrix) 
% indices : assign vectors (N x 10 matrix  in [numCV])
%% Options
%seed    :  seed for random vairables
%Without seed, we load preseparated indices from dataset/index
%             (recommended for comparisons)
%With seed,  we separate dataset to train/test with the given seed 

% load matrix files
tmp=load(['dataset/matfile/',dataname,'.mat']);
data=tmp.data;
target=tmp.target;

% load index files for train/test separation
if nargin <3
    %if seed is not given load preprocessed one
    tmp=load(['dataset/index/',num2str(numCV),'-fold/',dataname,'.mat']);
    indices=tmp.indices;
else
    rng(seed);
    indices=[];
    for i=1:10
        ind = crossvalind('Kfold',size(data,1),numCV);
        data=sparse(data);
        indices=[indices ind];
    end
end