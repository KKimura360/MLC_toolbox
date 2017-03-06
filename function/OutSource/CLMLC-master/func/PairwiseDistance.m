function [distance]=PairwiseDistance(x1,x2,A)
%%Input
%x1 instance 1 F-dimensional vec
%x2 instance 2 F-dimensional vec
% A ditsance metrix F x F matrix

distance=sqrt(((x1-x2)' * A)* (x1-x2));
