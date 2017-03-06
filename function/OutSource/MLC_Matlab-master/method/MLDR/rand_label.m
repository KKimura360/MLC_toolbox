function Y=rand_label(k, n)
% function Y=rand_label(k, n)
% This function randomly generates a label matrix for multi-class learning
%
%   Input:
%   k: the number of classes
%   n: the number of data samples
%
%   Output:
%  Y: k-by-n matrix. Y(i,j)=1 if xj belongs to class Ci, and Y(i,j)=0 otherwise. 
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

seq = randperm(n);
average_size = floor(n/k);
Y = zeros(k, n);

for i=1:k
    index_start = (i-1) * average_size + 1;
    if i~=k    
        index_end = i * average_size;
    else
        index_end = n;
    end
    seq_i = seq(index_start:index_end);
    Y(i, seq_i) = 1;
end

