function [distance]=AllDistance(X1,X2,A)
%%Input
%x1 NxF matrix
%x2 NxF matrix
% A ditsance metrix F x F matrix
%Output

distance=sqrt(((X1-X2) * A)* (X1-X2)');

