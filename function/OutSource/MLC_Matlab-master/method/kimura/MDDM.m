function [P L ] =MDDM(X,Y,K);
% X = feature x data 
% Y = label   x data

X=X';
Y=Y';
N=size(X,2);
%L=kerneltrick(Y');
L = Y'* Y;
H=eye(N) .*(1- (1/N)); 

tmp=(X * H) * L * (H *X');
[P,L]=eigs(tmp,K);

end
