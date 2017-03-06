function M = rowCenter(M, rowMean)
% function M = rowCenter(M, rowMean)
% This function centers each row of matrix M
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%
if nargin == 1
    rowMean = mean(M);
end
for r = 1:size(M,1)
    M(r, :) = M(r, :) - rowMean;
end
