function M = colCenter(M, colMean)
% function M = colCenter(M, colMean)
% This function centers each column of matrix M
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

if nargin == 1
    colMean = mean(M, 2);
end
for c = 1:size(M,2)
    M(:, c) = M(:, c) - colMean;
end
